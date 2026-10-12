#include "global.h"	// include first in any cpp
#include "Texture.h"
#include "Renderer.h"
#include "stb_image.h"	// implementation compiled once in App.cpp
#include <algorithm>
#include <cmath>

extern Renderer g_renderer;

CObjManager<CTexHandle, Texture> g_TextureManager;

void Texture::uploadPixels(VulkanContext& vulkanContext, uint32 width, uint32 height, const uint8_t* rgba, const std::string& debugName)
{
	// m_loadFromFile is already set by whichever of loadFromFile()/loadFromMemory() called us --
	// cache the decoded pixels only for the file-less case (see m_sourcePixels' own comment);
	// skipped for a real file since m_name/loadFromFile() already reloads it more compactly (JPG
	// bytes, not raw RGBA) and this can be a large amount of memory to keep around.
	//
	// The `rgba != (const uint8_t*)m_sourcePixels.data()` check guards a use-after-free when
	// rebuilding a file-less texture from its own cache (loadFromMemory(m_sourcePixels.data(), ...)):
	// without it, assign() would free m_sourcePixels' existing buffer (the very memory `rgba` points
	// into) before copying from it.
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

	// Traditional (non-heap) image view, for non-bindless binding -- see getImageView().
	const VkImageViewCreateInfo viewInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = m_image.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1},
	};
	VK_CHECK(vkCreateImageView(vulkanContext.device, &viewInfo, nullptr, &m_view));
	DBG_VK_NAME(m_view);
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
	ASSERT(data != nullptr, "Could not decode texture image!");
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

void Texture::deinit(VkDevice device, utils::ResourceAllocator& allocator)
{
	if (m_view != VK_NULL_HANDLE)
	{
		vkDestroyImageView(device, m_view, nullptr);
		m_view = VK_NULL_HANDLE;
	}
	allocator.destroyImageResource(m_image);
	// destroyImageResource() (extern/vk_framework.h) doesn't null out m_image itself -- reset it
	// here so isLoaded() (and this object generally) return to the same state as a freshly
	// constructed Texture, matching Buffer::deinit()'s existing "m_data = {}" convention.
	m_image = {};
}
