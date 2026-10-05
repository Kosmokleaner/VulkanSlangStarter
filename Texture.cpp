#include "global.h"	// include first in any cpp
#include "Texture.h"
#include "Renderer.h"
#include "AppendArena.h"
#include "stb_image.h"	// implementation compiled once in App.cpp
#include <algorithm>
#include <cmath>

extern Renderer g_renderer;

CObjManager<CTexHandle, Texture> g_TextureManager;

void GenerateNormalMapFromLuminance(const uint8_t* albedoRGBA, uint32 width, uint32 height, float strength, std::vector<uint32>& outNormalRGBA)
{
	outNormalRGBA.resize(size_t(width) * height);

	// Rec.709 luma from the raw (still sRGB-gamma-encoded) bytes directly -- this only needs to
	// track the albedo's own relative light/dark pattern as a height field, not a photometrically
	// exact decode, so skipping the sRGB->linear conversion here is fine (and cheaper).
	auto luminanceAt = [&](int32 x, int32 y) -> float
	{
		x = std::clamp(x, 0, (int32)width - 1);
		y = std::clamp(y, 0, (int32)height - 1);
		const uint8_t* p = albedoRGBA + (size_t(y) * width + x) * 4;
		return (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.0f;
	};

	for (uint32 y = 0; y < height; ++y)
	{
		for (uint32 x = 0; x < width; ++x)
		{
			// Central-difference gradient of the height field, standard "bump from heightmap"
			// tilt-the-normal-by-the-slope formula (z=1 is the untilted, flat-surface case).
			float dx = (luminanceAt((int32)x + 1, (int32)y) - luminanceAt((int32)x - 1, (int32)y)) * strength;
			float dy = (luminanceAt((int32)x, (int32)y + 1) - luminanceAt((int32)x, (int32)y - 1)) * strength;

			float3 n = float3(-dx, -dy, 1.0f);
			float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
			n = n / std::max(len, 1e-8f);

			auto toByte = [](float c) -> uint32 { return (uint32)std::clamp((c * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f); };
			uint32 r = toByte(n.x), g = toByte(n.y), b = toByte(n.z);

			outNormalRGBA[size_t(y) * width + x] = r | (g << 8) | (b << 16) | (0xFFu << 24);
		}
	}
}

void Texture::uploadPixels(VulkanContext& vulkanContext, uint32 width, uint32 height, const uint8_t* rgba, const std::string& debugName)
{
	// m_loadFromFile is already set by whichever of loadFromFile()/loadFromMemory() called us --
	// cache the decoded pixels only for the file-less case, so saveTextures() has something to
	// persist and loadScene() can rebuild this exact texture later (see m_sourcePixels' own
	// comment); skipped for a real file since m_name/loadFromFile() already reloads it more
	// compactly (JPG bytes, not raw RGBA) and this can be a large amount of memory to keep around
	// (e.g. a whole BSP level's worth of lightmaps) for something otherwise never read again.
	//
	// The `rgba != (const uint8_t*)m_sourcePixels.data()` check guards a real use-after-free:
	// loadScene() reloads a file-less texture by calling loadFromMemory() with this *same* vector's
	// own data() as the source (see its call site) -- without this check, assign() would free
	// m_sourcePixels' existing buffer (the very memory `rgba` points into) before copying from it,
	// reading freed memory (confirmed live as an access violation in vcruntime's allocator).
	if (!m_loadFromFile && rgba != (const uint8_t*)m_sourcePixels.data())
	{
		m_sourceWidth = width;
		m_sourceHeight = height;
		m_sourcePixels.assign((const uint32*)rgba, (const uint32*)rgba + size_t(width) * height);
	}

	// Hardware sRGB decode (free, done by the texture unit as part of normal filtering) instead of
	// a manual shader-side gamma decode per sample -- see m_isSRGB's own comment. Non-color data
	// (metallic-roughness, lightmaps) must stay UNORM/linear; the caller sets m_isSRGB=false before
	// calling loadFromFile()/loadFromMemory() for those.
	const VkFormat format = m_isSRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

	// Full mip chain down to 1x1, generated below via GPU blits -- without this, minified/angled
	// surfaces alias/shimmer instead of falling back to a properly-filtered lower-resolution level
	// (confirmed as a visible difference against a reference screenshot from a renderer that did
	// generate mips). VK_FORMAT_R8G8B8A8_UNORM/_SRGB are both mandatory-support formats for blit
	// src/dst + linear filtering on every conformant Vulkan implementation, so no capability check.
	uint32 mipLevels = 1;
	for (uint32 dim = std::max(width, height); dim > 1; dim >>= 1)
		++mipLevels;

	const VkImageCreateInfo imageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {width, height, 1},
		.mipLevels = mipLevels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		// TRANSFER_SRC (in addition to SAMPLED and the TRANSFER_DST createImageAndUploadData adds
		// itself) so each mip below can be blitted from the level above it.
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
	};

	const std::span dataSpan(rgba, size_t(width) * height * 4);
	m_image = vulkanContext.allocator.createImageAndUploadData(vulkanContext.cmd, dataSpan, imageInfo, VK_IMAGE_LAYOUT_GENERAL);
	utils::DebugUtil::getInstance().setObjectName(m_image.image, "Texture " + debugName);
	m_image.extent = {width, height};
	m_image.format = format;

	// createImageAndUploadData() already left the WHOLE image (every mip, via cmdInitImageLayout's
	// own VK_REMAINING_MIP_LEVELS range) in GENERAL layout with mip 0 populated -- GENERAL is a
	// valid blit src/dst layout, so no further layout transitions are needed, just a transfer
	// write->read barrier on each source mip between blits.
	for (uint32 level = 1; level < mipLevels; ++level)
	{
		const VkImageMemoryBarrier2 barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = m_image.image,
			.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, 1},
		};
		const VkDependencyInfo depInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
		vkCmdPipelineBarrier2(vulkanContext.cmd, &depInfo);

		const int32 srcW = (int32)std::max(1u, width >> (level - 1)), srcH = (int32)std::max(1u, height >> (level - 1));
		const int32 dstW = (int32)std::max(1u, width >> level), dstH = (int32)std::max(1u, height >> level);
		const VkImageBlit blit{
			.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1},
			.srcOffsets = {{0, 0, 0}, {srcW, srcH, 1}},
			.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
			.dstOffsets = {{0, 0, 0}, {dstW, dstH, 1}},
		};
		vkCmdBlitImage(vulkanContext.cmd, m_image.image, VK_IMAGE_LAYOUT_GENERAL, m_image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
	}

	// Traditional (non-heap) image view, for the ray tracing closest-hit shader's texture lookup
	// -- see Renderer::m_rtTextureViews' comment for why the heap can't be used there.
	const VkImageViewCreateInfo viewInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = m_image.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1},
	};
	VK_CHECK(vkCreateImageView(vulkanContext.device, &viewInfo, nullptr, &m_rtView));
	DBG_VK_NAME(m_rtView);

	// Register in the raster bindless descriptor heap; the returned index is also used, at the
	// same slot, for the ray tracing path's traditional texture array (capped, see
	// shaderio::kMaxRayTracingTextures -- textures beyond the cap keep a valid raster heap index
	// but are simply not reachable from the ray tracing path).
	m_heapIndex = g_renderer.m_descriptorHeap.addTexture(m_image.image, m_image.format, m_image.layout, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

	if (m_heapIndex < shaderio::kMaxRayTracingTextures)
		g_renderer.m_rtTextureViews[m_heapIndex] = m_rtView;

//	LOGI("Texture loaded: %s (%ux%u), descriptor heap index %u\n", debugName.c_str(), width, height, m_heapIndex);
}

void Texture::loadFromFile(VulkanContext& vulkanContext, const std::string& filename)
{
	m_name = filename;
	m_loadFromFile = true;

	int w = 0, h = 0, comp = 0;
	stbi_uc* data = stbi_load(filename.c_str(), &w, &h, &comp, 4);
//	ASSERT(data != nullptr, "Could not load texture image!");
	if(!data)
		return;	// failed

	uploadPixels(vulkanContext, uint32(w), uint32(h), data, filename);
	stbi_image_free(data);
}

void Texture::loadFromMemory(VulkanContext& vulkanContext, const uint8_t* encodedData, size_t encodedSize, const std::string& debugName)
{
	m_name.clear();
	m_loadFromFile = false;

	int w = 0, h = 0, comp = 0;
	stbi_uc* data = stbi_load_from_memory(encodedData, (int)encodedSize, &w, &h, &comp, 4);
	ASSERT(data != nullptr, "Could not decode glTF texture image!");
	uploadPixels(vulkanContext, uint32(w), uint32(h), data, debugName);
	stbi_image_free(data);
}

void Texture::loadFromMemory(VulkanContext& vulkanContext, const uint32_t* rgba, uint32 width, uint32 height, const std::string& debugName)
{
	m_name.clear();
	m_loadFromFile = false;

	ASSERT(rgba != nullptr);
	uploadPixels(vulkanContext, width, height, (const uint8_t*)rgba, debugName);
}

/*
void Texture::init(EventContext& context)
{
	if (m_loadFromFile)
		loadFromFile(vulkanContext, el.m_name);
}

void Texture::deinit(EventContext& context)
{
	if (m_rtView != VK_NULL_HANDLE)
	{
		vkDestroyImageView(context.device, m_rtView, nullptr);
		m_rtView = VK_NULL_HANDLE;
	}
	context.allocator.destroyImageResource(m_image);
}
*/

void Texture::deinit(VkDevice device, utils::ResourceAllocator& allocator)
{
	if (m_rtView != VK_NULL_HANDLE)
	{
		vkDestroyImageView(device, m_rtView, nullptr);
		m_rtView = VK_NULL_HANDLE;
	}
	allocator.destroyImageResource(m_image);
	// destroyImageResource() (extern/vk_framework.h) doesn't null out m_image itself -- reset it
	// here so isLoaded() (and this object generally) return to the same state as a freshly
	// constructed Texture, matching Buffer::deinit()'s existing "m_data = {}" convention.
	m_image = {};
}

void saveTextures(TokenAppendArena& arena)
{
	for (uint32 i = 0, count = g_TextureManager.GetElementCount(); i < count; ++i)
	{
		// todo: could be optimized to single function call
		auto hnd = g_TextureManager.GetHandleByIterator(i);
		auto& el = g_TextureManager.GetElementByIterator(i);

		TypedHandle dstKey(hnd, 2);	// 2 is Texture

		LOGI_AA_SAVE("Texture[%d] name=%s", dstKey.Index, el.m_name.c_str());

		// todo: use reflection
		{
			ATokenCmdNew cmd;
			cmd.dstKey = dstKey;
			arena.appendCmd(cmd);
		}
		{
			// todo: improve API !
			ATokenCmdBlob blob(el.m_name.size() + 1);
			arena.appendCmd(blob);

			size_t oldSize = arena.size();
			uint64 value0 = arena.appendString(el.m_name);
			size_t newSize = arena.size();
			ASSERT(newSize == oldSize + blob.count);

			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 0;
			cmd.valueOrIndexInArenaToValue = value0;
			arena.appendCmd(cmd);
			cmd.memberIndex = 1;
			cmd.valueOrIndexInArenaToValue = el.m_loadFromFile;
			arena.appendCmd(cmd);
			cmd.memberIndex = 5;
			cmd.valueOrIndexInArenaToValue = el.m_isSRGB;
			arena.appendCmd(cmd);
		}

		// A file-less texture (BSP lightmap, glTF-embedded image, ...) has no filename to reload
		// from -- save its raw decoded pixels instead, the same reason saveMeshes() saves
		// m_sourceIndices/m_sourceVertices for a procedural mesh. loadFromFile() (memberIndex 1
		// above) already covers the file case more compactly, so skip this there.
		if (!el.m_loadFromFile)
		{
			uint32 countPixels = (uint32)el.m_sourcePixels.size();
			uint32 sizePixels = countPixels * (uint32)sizeof(el.m_sourcePixels[0]);

			// todo: improve API !
			ATokenCmdBlob blob(sizeof(uint32) + sizePixels);
			arena.appendCmd(blob);

			size_t oldSize = arena.size();

			uint64 valuePixels = arena.appendBytes(&countPixels, sizeof(countPixels));
			// appendBytes() asserts sizeInBytes != 0 -- a file-less texture with no pixels actually
			// cached (shouldn't normally happen, but cheap to guard) would otherwise assert here.
			if (sizePixels)
				arena.appendBytes(el.m_sourcePixels.data(), sizePixels);

			size_t newSize = arena.size();
			ASSERT(newSize == oldSize + blob.count);

			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			// m_sourceWidth/m_sourceHeight are plain uint32s -- get(ATokenCmdSetMember*, uint32&)
			// reads valueOrIndexInArenaToValue as the value itself (see its definition,
			// AppendArena.cpp), not as an arena offset to dereference. Storing arena.appendBytes()'s
			// *return value* (an offset) here instead of the real width/height -- as this used to --
			// silently set m_sourceWidth/m_sourceHeight to that offset (a small, unrelated integer)
			// on load, which uploadPixels() then multiplied into a bogus width*height*4 byte span far
			// larger than m_sourcePixels actually holds: createStagingBuffer() reading that span read
			// off the end of the vector's allocation, an access violation confirmed live via a
			// VS call stack landing in ResourceAllocator::createStagingBuffer.
			cmd.memberIndex = 2;
			cmd.valueOrIndexInArenaToValue = el.m_sourceWidth;
			arena.appendCmd(cmd);
			cmd.memberIndex = 3;
			cmd.valueOrIndexInArenaToValue = el.m_sourceHeight;
			arena.appendCmd(cmd);
			cmd.memberIndex = 4;
			cmd.valueOrIndexInArenaToValue = valuePixels;
			arena.appendCmd(cmd);
		}
	}
}

void load(Texture& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping)
{
	// todo: reflection with backward compatible loading
	if (cmd->memberIndex == 0) get(cmd, inoutRange, ref.m_name);
	else if (cmd->memberIndex == 1) get(cmd, inoutRange, ref.m_loadFromFile);
	// memberIndex 2/3/4: raw pixel data for a file-less texture (BSP lightmap, glTF-embedded
	// image, ...) -- see Texture::m_sourcePixels' own comment. Only present in the saved file
	// when saveTextures() found !m_loadFromFile, same as mesh IB/VB (RenderComponent.cpp).
	else if (cmd->memberIndex == 2) get(cmd, inoutRange, ref.m_sourceWidth);
	else if (cmd->memberIndex == 3) get(cmd, inoutRange, ref.m_sourceHeight);
	else if (cmd->memberIndex == 4) getVector(cmd, inoutRange, ref.m_sourcePixels);
	// Not present in an older save file (defaults to m_isSRGB's own default, true) -- see
	// Texture.h's m_isSRGB comment.
	else if (cmd->memberIndex == 5) get(cmd, inoutRange, ref.m_isSRGB);
	else ASSERT(0);
}