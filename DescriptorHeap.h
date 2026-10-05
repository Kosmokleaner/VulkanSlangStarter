#pragma once
#include "vulkanBase.h"
#include "EventManager.h"

class DescriptorHeap
{
public:
	utils::Buffer m_resourceHeapBuffer;
	uint32 m_maxTextures = 10000;           // Maximum textures allowed in the application
	VkDeviceSize m_resourceHeapSize{};
	VkDeviceSize m_resourceReservedOffset{};  // Byte offset where the resource reserved range starts
	VkDeviceSize m_resourceReservedSize{};    // Size of the mandatory resource reserved range

	//
	void init(VkPhysicalDevice physicalDevice, VkCommandBuffer cmd);
	//
	void deinit(EventContext& context);

	// intentionally not doing samplers here as we have m_samplerPool

	//
	// aspectMask defaults to VK_IMAGE_ASPECT_COLOR_BIT; pass VK_IMAGE_ASPECT_DEPTH_BIT for a depth
	// image's own view (e.g. RenderTarget's depth buffer, read back as a Texture2D for TAA's
	// raster-path reprojection) -- VUID-VkImageDescriptorInfoEXT-pView-11430 requires it match the
	// image's own aspect, a color aspect view against a depth/stencil image is invalid.
	// mipCount > 1 registers a view spanning the image's whole generated mip chain (see Texture.cpp's
	// own mip generation) instead of just mip 0.
	uint32 addTexture(VkImage image, VkFormat format, VkImageLayout layout, VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, uint32 mipCount = 1);
	// Same as addTexture(), but registers a VK_IMAGE_VIEW_TYPE_CUBE view spanning all 6 layers and
	// mipCount mip levels instead of addTexture()'s single-layer, single-mip VK_IMAGE_VIEW_TYPE_2D
	// -- for an IBL environment cubemap (see IBLEnvironment.h), sampled bindlessly the same way
	// (TextureCube.Handle(uint2(index, 0))) once registered.
	uint32 addCubeTexture(VkImage image, VkFormat format, VkImageLayout layout, uint32 mipCount);
	//
	uint32 addStorageImage(VkImage image, VkFormat format);
	//
	uint32 addTLAS(VkAccelerationStructureKHR tlas);
	//
	// Overwrites an existing slot in place instead of appending a new one -- for a resource that gets
	// destroyed/recreated (e.g. on resize) whose heap index needs to stay the same *and* never leave a
	// stale entry pointing at an already-destroyed VkImage behind: calling addTexture() again on every
	// resize instead of this appended one dangling m_resources entry per resize (the old, now-freed
	// image), which the next build() then tried to create an image view for -- crashing with
	// "vkWriteResourceDescriptorsEXT(): ... Invalid VkImage Object" the moment two resizes happened
	// before the app ever got a chance to use the first one (routine: see EnsureGBufferCapacity's own
	// comment on ImGui's docking layout settling). @param index must be one addTexture() already
	// returned.
	void updateTexture(uint32 index, VkImage image, VkFormat format, VkImageLayout layout, VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, uint32 mipCount = 1);
	// Same as updateTexture(), for a slot addStorageImage() returned.
	void updateStorageImage(uint32 index, VkImage image, VkFormat format);
	//
	// Drops every previously-added resource (currently only ever textures -- addStorageImage()/
	// addTLAS() have no callers yet) so the next build() starts from an empty set. Needed before
	// re-adding textures whose old VkImages are about to be (or already were) destroyed -- e.g.
	// reloading a scene discards and recreates every Texture -- since build() would otherwise
	// create image views against freed images.
	void resetResources() { m_resources.clear(); }
	//
	void build(VkDevice device, utils::ResourceAllocator& allocator, VkCommandBuffer cmd);
	//
	void bind(VkCommandBuffer cmd);

private: // ===========================================
	VkDeviceSize m_imageDescriptorSize = {};
	VkDeviceSize m_heapAlignment = {};

	struct ResourceDesc
	{
		VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;

		VkImage image = VK_NULL_HANDLE;
		VkFormat format = VK_FORMAT_UNDEFINED;
		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		// VK_IMAGE_VIEW_TYPE_2D (layerCount=1, mipCount=1) for everything addTexture()/
		// addStorageImage() registers; VK_IMAGE_VIEW_TYPE_CUBE (layerCount=6, mipCount=however many
		// the specular map has) for addCubeTexture().
		VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
		uint32 layerCount = 1;
		uint32 mipCount = 1;
		VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

		VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
	};

	std::vector<ResourceDesc> m_resources;
};
