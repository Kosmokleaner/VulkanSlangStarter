#include "global.h"	// include first in any cpp
#include "Renderer.h"
#include "Buffer.h"
#include "Scene.h"
#include "ShaderCache.h"
#include "RenderComponent.h"
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

	{
		VulkanContext vulkanContext(m_context.getDevice(), m_context.getPhysicalDevice(), m_allocator, cmd, m_context.getGraphicsQueue().queue, m_transientCmdPool);

		// Procedural, not loaded from disk: this used to be resources/ReplaceMe.png, but that file
		// doesn't exist in the repo -- Texture::loadFromFile() fails silent when stbi_load() returns
		// null (see its own "return; // failed" comment), so uploadPixels()/addTexture() never ran
		// and this texture never actually claimed descriptor-heap index 0 at all. The assert below
		// didn't catch it either -- hdn.Index is g_TextureManager's own handle index, unrelated to
		// m_heapIndex (the descriptor-heap slot addTexture() returns), so it stayed trivially true.
		// Whatever the next addTexture() call happened to be (e.g. a G-buffer/render-target/TAA
		// history registration -- none of which exist yet when this runs, but do the moment any of
		// them touches the heap before a real scene texture does) silently claimed slot 0 instead,
		// permanently breaking Shader::updateRayDescriptorSet's textureViews[0] guard (confirmed live
		// as vkCmdTraceRaysKHR's VUID-vkCmdTraceRaysKHR-None-08114, "g_tlas ... has never been
		// updated" -- that guard returns early forever once slot 0 stops being a real Texture2D). A
		// tiny in-memory checkerboard can't go missing, so this no longer depends on any file at all.
		const uint32 kMagenta = 0xFFFF00FFu; // little-endian RGBA8 (see Texture::loadFromMemory's own byte-order comment): R=FF G=00 B=FF A=FF
		const uint32 kBlack = 0xFF000000u;   // R=00 G=00 B=00 A=FF
		const uint32 pixels[4] = { kMagenta, kBlack, kBlack, kMagenta };

		Texture tex0;
		tex0.loadFromMemory(vulkanContext, pixels, 2, 2, "ReplaceMe (procedural fallback)");
		tex0.isEngineTexture = true;
		CTexHandle hdn = g_TextureManager.push_back(tex0);
		ASSERT(hdn.Index == 0);	// replaceme texture is fallback with index 0
	}
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
