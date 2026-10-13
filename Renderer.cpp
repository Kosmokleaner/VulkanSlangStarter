// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0
// Portions derived from src/minimal_latest.cpp in https://github.com/nvpro-samples/vk_minimal_latest,
// modified by Martin Mittring.

#include "global.h"	// include first in any cpp
#include "Renderer.h"
#include "Buffer.h"
#include "ShaderCache.h"
#include "Texture.h"

#include "vulkanBase.h"
#include <glm/gtc/type_ptr.hpp> //  glm::value_ptr



Renderer::Renderer()
{

}

void Renderer::init(VkPhysicalDevice physicalDevice, VkCommandBuffer cmd)
{
	g_ShaderCache.Load();
	m_descriptorHeap.init(physicalDevice, cmd);
}

void Renderer::deinit(VkDevice device, utils::ResourceAllocator& allocator)
{
	ASSERT(device);

	EventContext context{ device, allocator };
	m_descriptorHeap.deinit(context);

	for (TlasSlot& slot : m_tlasSlots)
	{
		if (slot.tlas != VK_NULL_HANDLE)
		{
			vkDestroyAccelerationStructureKHR(device, slot.tlas, nullptr);
			slot.tlas = {};
		}

		slot.tlasInstancesBuffer.deinit(context);
		slot.tlasBuffer.deinit(context);
		slot.scratch.deinit(context);
	}

	g_ShaderCache.Save();
}

void Renderer::reloadShaders()
{
	Log("\n");

	VkDevice device = m_context.getDevice();
	VK_CHECK(vkDeviceWaitIdle(device));

	EventContext context{ m_context.getDevice(), m_allocator };
	g_EventManager.callReloadShaders(context);

}

void Renderer::createTransientCommandPool()
{
	m_transientCmdPool = utils::createCommandPool(m_context.getDevice(), m_context.getGraphicsQueue().familyIndex,
		VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	DBG_VK_NAME(m_transientCmdPool);
}
