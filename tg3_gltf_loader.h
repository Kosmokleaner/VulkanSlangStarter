#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "tinygltf/tiny_gltf_v3.h"
#include "ObjManager.h"	// CObjHandle

struct GLTFVertex
{
	float3 position{};
	float3 normal{};
	float4 tangent{}; // remove if unused
	float2 uv{};
};

// One glTF primitive's worth of the combined indices below, plus which model.materials[] entry
// (if any) it was authored with.
struct GltfSubMeshRange
{
	uint32_t indexStart = 0;
	uint32_t indexCount = 0;
	// index into model.materials[], -1 if the primitive had none
	int32_t materialIndex = -1;
};

struct GltfMeshData
{
	std::vector<GLTFVertex> vertices;
	std::vector<uint32_t> indices;
	// one entry per glTF primitive across every node in the loaded file, in the same order they
	// were appended to indices
	std::vector<GltfSubMeshRange> subMeshes;

	void addAABB(float3 min_xyz, float3 max_xyz, float3 normal)
	{
		uint32_t start = (uint32_t)vertices.size();

		float3 aabbPos[] = {
			{min_xyz.x, min_xyz.y, min_xyz.z}, {max_xyz.x, min_xyz.y, min_xyz.z},
			{max_xyz.x, max_xyz.y, min_xyz.z}, {min_xyz.x, max_xyz.y, min_xyz.z},
			{min_xyz.x, min_xyz.y, max_xyz.z}, {max_xyz.x, min_xyz.y, max_xyz.z},
			{max_xyz.x, max_xyz.y, max_xyz.z}, {min_xyz.x, max_xyz.y, max_xyz.z}
		};
		for (int i = 0; i < 8; ++i)
		{
			GLTFVertex v;
			v.position = aabbPos[i];
			v.normal = normal;
			vertices.push_back(v);
		}

		// 36 indices for 12 triangles (2 triangles per face)
		uint32_t aabbIdx[] = {
			4, 5, 6,  4, 6, 7, // Front face
			1, 0, 3,  1, 3, 2, // Back face
			0, 4, 7,  0, 7, 3, // Left face
			5, 1, 2,  5, 2, 6, // Right face
			7, 6, 2,  7, 2, 3, // Top face
			0, 1, 5,  0, 5, 4  // Bottom face
		};
		for (int i = 0; i < 36; ++i)
			indices.push_back(start + aabbIdx[i]);
	}

	void addAxis(float3 origin, float3 size, float thickness)
	{
		float3 t = float3(thickness, thickness, thickness);
		addAABB(origin - t, float3(size.x, 0, 0) + t, float3(1, 0, 0));	// R X
		addAABB(origin - t, float3(0, size.y, 0) + t, float3(0, 1, 0));	// G Y
		addAABB(origin - t, float3(0, 0, size.z) + t, float3(0, 0, 1));	// B Z
	}
};


inline int tg3_find_attribute(const tg3_primitive& p, const char* name)
{
	for (uint32_t i = 0; i < p.attributes_count; i++)
	{
		const auto& a = p.attributes[i];

		if (a.key.len == strlen(name) &&
			memcmp(a.key.data, name, a.key.len) == 0)
			return a.value;
	}

	return -1;
}


inline const uint8_t* tg3_accessor_ptr(
	const tg3_model& model,
	int accessorIndex,
	uint32_t& stride)
{
	const tg3_accessor& a = model.accessors[accessorIndex];

	const tg3_buffer_view& view =
		model.buffer_views[a.buffer_view];

	const tg3_buffer& buffer =
		model.buffers[view.buffer];

	stride = view.byte_stride;

	if (stride == 0)
	{
		switch (a.type)
		{
		case TG3_TYPE_SCALAR:
			stride = a.component_type == TG3_COMPONENT_TYPE_UNSIGNED_BYTE ? 1 :
				a.component_type == TG3_COMPONENT_TYPE_UNSIGNED_SHORT ? 2 :
				4;
			break;

		case TG3_TYPE_VEC2:
			stride = 2 * sizeof(float);
			break;

		case TG3_TYPE_VEC3:
			stride = 3 * sizeof(float);
			break;

		case TG3_TYPE_VEC4:
			stride = 4 * sizeof(float);
			break;
		}
	}

	return buffer.data.data +
		view.byte_offset +
		a.byte_offset;
}


template<typename T>
inline T tg3_read(
	const tg3_model& model,
	int accessor,
	uint32_t index)
{
	uint32_t stride;
	const uint8_t* ptr =
		tg3_accessor_ptr(model, accessor, stride);

	return *reinterpret_cast<const T*>(ptr + index * stride);
}

inline uint32_t tg3_read_index(
	const tg3_model& model,
	const tg3_accessor& a,
	uint32_t index)
{
	uint32_t stride;
	const uint8_t* ptr = tg3_accessor_ptr(model, (int)(&a - model.accessors), stride);

	ptr += index * stride;

	switch (a.component_type)
	{
	case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
		return *reinterpret_cast<const uint8_t*>(ptr);

	case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
		return *reinterpret_cast<const uint16_t*>(ptr);

	case TG3_COMPONENT_TYPE_UNSIGNED_INT:
		return *reinterpret_cast<const uint32_t*>(ptr);
	}

	return 0;
}

bool tg3_load_meshes(const tg3_model& model, GltfMeshData& out);

struct VulkanContext;

// Creates one engine Material (Material.h) per model.materials[] entry, registered in
// g_MaterialManager, and appends its handle to outMaterialHandles (index-aligned with
// model.materials[], i.e. outMaterialHandles[i] is the handle for model.materials[i]). Populates
// albedoColor/roughness/metallic/emissive plus the base color texture (decoded and uploaded via
// Texture.h/g_TextureManager, see Material::m_hndTexture) -- everything else a glTF material can
// carry (metallic-roughness/normal/occlusion/emissive textures, alpha mode, ...) is left at
// Material's defaults for the caller to wire up manually later. Does not touch
// tg3_primitive::material (per-primitive material assignment) or any Scene/rendering state --
// this only creates the Material objects and hands back their handles.
bool tg3_load_materials(const tg3_model& model, VulkanContext& vulkanContext, std::vector<CMatHandle>& outMaterialHandles);

