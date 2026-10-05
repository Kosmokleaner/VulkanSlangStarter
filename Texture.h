#pragma once
#include "vulkanBase.h"
#include "ObjManager.h"
//#include "EventManager.h"
#include <string>
#include <vector>

struct VulkanContext;

// A single loaded texture image, registered both in the raster path's bindless descriptor heap
// (g_renderer.m_descriptorHeap) and, at the same index, in the ray tracing path's traditional
// texture array (g_renderer.m_rtTextureViews) -- see Renderer.h's m_rtTextureViews comment for why
// ray tracing can't just use the heap. Managed the same way as Material: stored by value in
// g_TextureManager, referenced elsewhere via the CObjHandle AddElement() returns.
class Texture //todo : public IAppInit
{
	utils::ImageResource m_image;
	VkImageView m_rtView = VK_NULL_HANDLE;
	uint32 m_heapIndex = 0;

public: // todo 
	// if true, use m_name as filename to load the textures
	bool m_loadFromFile = false;

public:
	// filename if loadFromFile() was called
	std::string m_name;
	// todo: refactor as flags, true: not saves or removed on load
	bool isEngineTexture = false;

	// true (the default) for color data -- albedo/diffuse/base-color, decoded gamma->linear for
	// free by the GPU's fixed-function texture sampler (see uploadPixels' format choice) instead of
	// spending shader ALU on it per-sample. Set to false by the caller BEFORE loadFromFile()/
	// loadFromMemory() for non-color data (metallic-roughness maps, lightmaps -- see their own call
	// sites) that must stay linear/undecoded. Persisted (saveTextures()/AppendArena.cpp) so a
	// restored scene's textures upload with the same format they were authored with.
	bool m_isSRGB = true;

	// Raw RGBA8 pixel data, cached only when !m_loadFromFile (see uploadPixels()) -- there's no
	// filename to reload from for a texture built at runtime (e.g. a BSP level's lightmaps, decoded
	// from the .bsp file's own embedded lightmap lump, see 3DBSPLevel.cpp; or a glTF material's
	// embedded image). Saved/restored by saveTextures()/AppendArena.cpp so loadScene() can rebuild
	// these via loadFromMemory() instead of leaving them blank -- the texture equivalent of
	// MeshComponent::m_sourceIndices/m_sourceVertices for a procedural (file-less) mesh.
	std::vector<uint32> m_sourcePixels;
	uint32 m_sourceWidth = 0;
	uint32 m_sourceHeight = 0;

	// @param filename e.g. "resources/image1.jpg", relative to the repo root
	void loadFromFile(VulkanContext& vulkanContext, const std::string& filename);
	// @param encodedData/encodedSize still-encoded (e.g. PNG/JPEG) image bytes, such as a glTF
	// image's embedded bufferView contents -- decoded here via stb_image, same as loadFromFile.
	void loadFromMemory(VulkanContext& vulkanContext, const uint8_t* encodedData, size_t encodedSize, const std::string& debugName);

	// @param rgba must not be 0. One pixel per uint32, but NOT a 0xAARRGGBB numeric value --
	// byte 0 (bits 0..7) must be red, byte 1 green, byte 2 blue, byte 3 alpha, matching this
	// platform's little-endian memory order for VK_FORMAT_R8G8B8A8_UNORM (see uploadPixels).
	// Called "ARGB" before, which reads as a 0xAARRGGBB hex value -- on this little-endian
	// platform that would put blue in byte 0 and red in byte 2, an R/B swap once reinterpreted
	// as raw bytes (see the one caller, 3DBSPLevel.cpp's lightmap loading, for how to pack this).
	void loadFromMemory(VulkanContext& vulkanContext, const uint32_t* rgba, uint32 width, uint32 height, const std::string& debugName);

	// destroys the GPU image/view
	void deinit(VkDevice device, utils::ResourceAllocator& allocator);

	// interface IAppInit ====================================

	// index into both the raster descriptor heap and (if < shaderio::kMaxRayTracingTextures) the
	// ray tracing traditional texture array -- this is the value stored in MaterialData::textureId
	uint32 getHeapIndex() const { return m_heapIndex; }

	// False for a never-loaded Texture, or one deinit() already tore down (deinit() resets
	// m_image, same "return to a freshly-constructed state" contract as Buffer::deinit()) -- lets
	// a caller holding onto a Texture across a deinit()/reload cycle (an engine texture surviving
	// loadScene()'s teardown sweep, e.g.) tell whether it still needs re-uploading.
	bool isLoaded() const { return m_image.image != VK_NULL_HANDLE; }

private:
	void uploadPixels(VulkanContext& vulkanContext, uint32 width, uint32 height, const uint8_t* rgba, const std::string& debugName);
};

extern CObjManager<CTexHandle, Texture> g_TextureManager;
