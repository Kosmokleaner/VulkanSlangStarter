#pragma once
#include "vulkanBase.h"
#include "ObjManager.h"
//#include "EventManager.h"
#include <string>
#include <vector>

struct VulkanContext;

// A single loaded texture image (full mip chain, VK_IMAGE_LAYOUT_GENERAL). Non-bindless shaders
// sample it through getImageView() plus a VkSampler in a classic descriptor set, see
// Shader::cmdBindTextures(). Stored by value in g_TextureManager, referenced elsewhere via the
// CTexHandle push_back() returns.
class Texture //todo : public IAppInit
{
	utils::ImageResource m_image;
	// classic (non-heap) view over all mips, for non-bindless binding
	VkImageView m_view = VK_NULL_HANDLE;

public: // todo 
	// if true, use m_name as filename to load the textures
	bool m_loadFromFile = false;

public:
	// filename if loadFromFile() was called
	std::string m_name;

	// true (the default) for color data -- albedo/diffuse/base-color, decoded gamma->linear for
	// free by the GPU's fixed-function texture sampler (see uploadPixels' format choice) instead of
	// spending shader ALU on it per-sample. Set to false by the caller BEFORE loadFromFile()/
	// loadFromMemory() for non-color data (e.g. metallic-roughness maps, lightmaps) that must stay
	// linear/undecoded.
	bool m_isSRGB = true;

	// Raw RGBA8 pixel data, cached only when !m_loadFromFile (see uploadPixels()) -- there's no
	// filename to reload from for a texture built at runtime, so this is what loadFromMemory() can
	// rebuild it from. Nothing reads it yet.
	std::vector<uint32> m_sourcePixels;
	uint32 m_sourceWidth = 0;
	uint32 m_sourceHeight = 0;

	// @param filename e.g. "resources/image1.jpg", relative to the repo root
	void loadFromFile(VulkanContext& vulkanContext, const std::string& filename);
	// @param encodedData/encodedSize still-encoded (e.g. PNG/JPEG) image bytes already in memory
	// -- decoded here via stb_image, same as loadFromFile.
	void loadFromMemory(VulkanContext& vulkanContext, const uint8_t* encodedData, size_t encodedSize, const std::string& debugName);

	// @param rgba must not be 0. One pixel per uint32, but NOT a 0xAARRGGBB numeric value --
	// byte 0 (bits 0..7) must be red, byte 1 green, byte 2 blue, byte 3 alpha, matching this
	// platform's little-endian memory order for VK_FORMAT_R8G8B8A8_UNORM (see uploadPixels), i.e.
	// 0xAABBGGRR as a numeric value.
	void loadFromMemory(VulkanContext& vulkanContext, const uint32_t* rgba, uint32 width, uint32 height, const std::string& debugName);

	// destroys the GPU image/view
	void deinit(VkDevice device, utils::ResourceAllocator& allocator);

	// False for a never-loaded Texture (including a failed loadFromFile()), or one deinit() already
	// tore down (deinit() resets m_image, same "return to a freshly-constructed state" contract as
	// Buffer::deinit()).
	bool isLoaded() const { return m_image.image != VK_NULL_HANDLE; }

	// for non-bindless binding (VkDescriptorImageInfo::imageView, layout VK_IMAGE_LAYOUT_GENERAL),
	// VK_NULL_HANDLE if !isLoaded()
	VkImageView getImageView() const { return m_view; }

private:
	void uploadPixels(VulkanContext& vulkanContext, uint32 width, uint32 height, const uint8_t* rgba, const std::string& debugName);
};

extern CObjManager<CTexHandle, Texture> g_TextureManager;
