// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0
// Portions derived from src/minimal_latest.cpp in https://github.com/nvpro-samples/vk_minimal_latest,
// modified by Martin Mittring.

#pragma once

#include "vulkanBase.h"
#include "Buffer.h"
#include "DescriptorHeap.h"

// Everything a one-off GPU upload (e.g. Texture::loadFromFile) needs. cmd is a single-time command
// buffer (allocated from transientCmdPool) that the caller submits after these functions return.
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

	// Must be >= utils::Swapchain::kPreferredFramesInFlight (extern/vk_framework.h). Sizes the
	// per-frame-in-flight resources, e.g. Shader::m_descSet and m_tlasSlots.
	static constexpr uint32 kMaxFramesInFlight = 2;

	// Ray tracing TLAS resources, duplicated per frame-in-flight slot (m_tlasSlots) so that
	// rebuilding it every frame never has to wait for a previous frame's ray tracing dispatch to
	// finish reading the slot it's still using -- the same double-buffering idea as App::m_frameData.
	// Not built yet (ray tracing isn't implemented), only released in deinit().
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

	void init(VkPhysicalDevice physicalDevice, VkCommandBuffer cmd);

	// @param device must not be 0
	void deinit(VkDevice device, utils::ResourceAllocator& allocator);

	void reloadShaders();

	// Create a command pool for short lived operations (one-off uploads, image layout
	// transitions, ...). TRANSIENT_BIT hints the driver that command buffers from this pool are
	// short-lived.
	void createTransientCommandPool();
};
