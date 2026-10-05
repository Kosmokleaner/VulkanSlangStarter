#pragma once

class Scene;

#include "vulkanBase.h"
#include "Buffer.h"
#include "DescriptorHeap.h"

// no command pool, the assumption is after functions using this context the cmd is submitted
struct VulkanContext
{
	VkDevice device;
	VkPhysicalDevice physicalDevice;
	//	Buffer& scratchMemory;
	utils::ResourceAllocator& allocator;
	// allocated from a command pool
	VkCommandBuffer cmd;
	VkQueue queue;	// m_context.getGraphicsQueue().queue
	VkCommandPool transientCmdPool;
};


class Renderer
{
public: // todo
	// The Vulkan context; owned here so it can be reached as g_renderer.m_context from anywhere
	// without threading it through every call site.
	utils::Context m_context;
	// The VMA allocator; same reasoning -- reachable as g_renderer.m_allocator.
	utils::ResourceAllocator m_allocator;

	// Must match Shader::kMaxFramesInFlight / utils::Swapchain::kPreferredFramesInFlight
	// (extern/vk_framework.h).
	static constexpr uint32 kMaxFramesInFlight = 2;

	// Everything createTopLevelAS needs to rebuild the TLAS, duplicated per frame-in-flight slot
	// (m_tlasSlots) so that rebuilding it every frame never has to wait for a previous frame's ray
	// tracing dispatch to finish reading the slot it's still using -- the same double-buffering
	// idea as App::m_frameData, reusing its existing per-slot GPU-completion wait instead
	// of adding any new fence/barrier.
	struct TlasSlot
	{
		// buffer with per-instance data (transform, instance id, BLAS reference, flags, mask)
		Buffer tlasInstancesBuffer;
		// buffer with the built TLAS tree data
		Buffer tlasBuffer;
		// scratch memory needed during TLAS construction
		Buffer scratch;
		// Top Level Acceleration structure built over tlasBuffer
		VkAccelerationStructureKHR tlas = {};

		TlasSlot() : tlasInstancesBuffer(EF_Type_Unknown), tlasBuffer(EF_Type_Unknown), scratch(EF_Type_Unknown) {}
	};
	TlasSlot m_tlasSlots[kMaxFramesInFlight];

	VkCommandPool m_transientCmdPool{};

public:
	Renderer();

	DescriptorHeap m_descriptorHeap;

	// Traditional (non-heap) binding for the ray tracing closest-hit shader's texture lookup --
	// see Shader::updateRayDescriptorSet's comment for why the heap can't be used there. Index i
	// here is the same index Texture::getHeapIndex() returns for the raster descriptor heap
	// (both are assigned together in Texture::uploadPixels), so MaterialData::textureId indexes
	// either array correctly.
	VkImageView m_rtTextureViews[shaderio::kMaxRayTracingTextures]{};

#if USE_RAY_TRACING
	// Device-wide ray tracing pipeline properties (shader group handle size/alignment, etc.),
	// queried once in init(). Shared by every ray tracing Shader instance's SBT layout math
	// (Shader::dispatchRays) -- it's a property of the physical device, not of any one pipeline.
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProps{};
#endif

	void init(VkPhysicalDevice physicalDevice, VkCommandBuffer cmd);

#if USE_RAY_TRACING
	// Rebuilds the TLAS for frameSlot from the scene's current instance transforms. Safe to call
	// every frame without stalling the GPU: see TlasSlot's comment for why.
	// @param frameSlot: m_swapchain.getFrameResourceIndex(), same slot m_frameData is indexed by.
	void createTopLevelAS(VulkanContext& vulkanContext, Scene& scene, uint32 frameSlot);
#endif

	// @param device must not be 0
	void deinit(VkDevice device, utils::ResourceAllocator& allocator);

	void reloadShaders();

	// Create a command pool for short lived operations (one-off uploads, image layout
	// transitions, ...). TRANSIENT_BIT hints the driver that command buffers from this pool are
	// short-lived.
	void createTransientCommandPool();
};
