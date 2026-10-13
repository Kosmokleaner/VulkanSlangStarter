// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0
// Portions derived from src/minimal_latest.cpp in https://github.com/nvpro-samples/vk_minimal_latest,
// modified by Martin Mittring.

#pragma once

#include "vulkanBase.h"
#include "EventManager.h"

namespace shaderio {  // Shader IO namespace -- shared layout between C++ and shaders
	using namespace glm;  // GLSL-style types without the glm:: prefix inside the namespace
#include "shaders/shader_io.h"
}  // namespace shaderio


// see "struct Vertex" in shader code (shader_io.h)
struct Vertex : public shaderio::Vertex
{
	/*--
	 * The binding description tells the pipeline at which rate to load data from
	 * memory throughout the vertices. Returned by const-reference to a static
	 * constexpr array so we don't reconstruct it on every call.
	-*/
	static const auto& getBindingDescription()
	{
		static constexpr auto kBindings = std::to_array<VkVertexInputBindingDescription>(
			{ {.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX} });
		return kBindings;
	}

	/*--
	 * The attribute descriptions describe how to extract a vertex attribute from
	 * a chunk of vertex data originating from a binding description.
	 * See in the vertex shader how the location is used to access the data.
	-*/
	static const auto& getAttributeDescriptions()
	{
		static constexpr auto kAttributes = std::to_array<VkVertexInputAttributeDescription2EXT>(
			{
			 {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = shaderio::LVPosition, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = uint32(offsetof(Vertex, position))},
			 {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = shaderio::LVNormal, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = uint32(offsetof(Vertex, normal))},
			 {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = shaderio::LVTangent, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = uint32(offsetof(Vertex, tangent))},
			 {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = shaderio::LVTexCoord, .format = VK_FORMAT_R32G32_SFLOAT, .offset = uint32(offsetof(Vertex, texCoord))},
			 {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = shaderio::LVLightmapUV, .format = VK_FORMAT_R32G32_SFLOAT, .offset = uint32(offsetof(Vertex, lightmapUV))},
//			 {.location = shaderio::LVColor, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = uint32(offsetof(Vertex, color))},
			});
		return kAttributes;
	}
};

class Buffer : public IAppInit
{
public: // todo
	// internals
	utils::Buffer m_data;
	// in bytes
	size_t m_size = {};
	// in bytes
	uint32 m_stride = {};

	void deinitInternal();

public:
	enum EBufferType
	{
		eBT_IndexBuffer,
		eBT_VertexBuffer,
		eBT_ShaderBindingTable,
	};

	Buffer(EEventFlags flags);

	// destructor
	~Buffer();

	// with CPU buffer to copy
	// needs to be outside vkCmdBeginRendering/vkCmdEndRendering 
	// @param data must not be 0
	// @param size in bytes
	// @param stride in bytes
	void initRaw(utils::ResourceAllocator& allocator, VkCommandBuffer& cmd, const void* data, size_t count, uint32 stride, EBufferType type);

	// without CPU buffer to copy (VMA-backed, like initRaw -- no manual memory management,
	// no raw VkDeviceMemory handle to track). @param minAlignment: 0 for VMA's default, or a
	// specific requirement such as minAccelerationStructureScratchOffsetAlignment.
	void init(utils::ResourceAllocator& allocator, VkDeviceSize size, VkBufferUsageFlags2 usage, VkDeviceSize minAlignment = 0);

	// interface IAppInit
	virtual void init(EventContext& context);
	// safe to call every frame (e.g. rebuilding this Buffer each frame): destruction is deferred
	// until the GPU has actually finished with it, see DeferredRelease.h
	virtual void deinit(EventContext& context);

	// ---------------

	// with CPU buffer to copy
	template <class T>
	void initSpan(utils::ResourceAllocator& allocator, VkCommandBuffer& cmd, std::span<T> elements)
	{
		initRaw(allocator, cmd, (const void*)elements.data(), elements.size(), (uint32)sizeof(T), eBT_VertexBuffer);
	}
};
