#include "global.h"	// include first in any cpp
#include "Buffer.h"
#include "DeferredRelease.h"

Buffer::Buffer(EEventFlags flags)
	: IAppInit(flags)
{
}

Buffer::~Buffer()
{
	// forgot to call deinit()
	ASSERT(!m_data.address);
	ASSERT(!m_data.allocation);
	ASSERT(!m_data.buffer);
	ASSERT(!m_size);
	ASSERT(!m_stride);
}

void Buffer::initRaw(utils::ResourceAllocator& allocator, VkCommandBuffer& cmd, const void* data, size_t count, uint32 stride, EBufferType type)
{
	deinitInternal();

	ASSERT(data);

	VkBufferUsageFlagBits2 flags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;

	if (type == eBT_IndexBuffer)
	{
		flags |= VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT;
		flags |= VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;	// for ray tracing
	}
	else if (type == eBT_VertexBuffer)
	{
		flags |= VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT;
		flags |= VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;	// for ray tracing
	}
	else if (type == eBT_ShaderBindingTable)
	{
		flags = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	}
	else ASSERT(0);
	//	else if (type == eBT_Scratch)
	//	{
	//		flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	//	}

	const VkDeviceSize bufferSize = stride * count;

	m_data = allocator.createBufferAndUploadData(cmd, data, bufferSize, flags);
	DBG_VK_NAME2(m_data.buffer, "Buffer");

	m_size = bufferSize;
	m_stride = stride;
}

void Buffer::init(utils::ResourceAllocator& allocator, VkDeviceSize size, VkBufferUsageFlags2 usage, VkDeviceSize minAlignment)
{
	deinitInternal();

	m_data = allocator.createBuffer(size, usage | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, {}, minAlignment);
}

void Buffer::deinitInternal()
{
	g_deferredRelease.Release(m_data);

	m_data = {};
	m_size = {};
	m_stride = {};
}

void Buffer::init(EventContext& context)
{
}

void Buffer::deinit(EventContext& context)
{
	deinitInternal();
}
