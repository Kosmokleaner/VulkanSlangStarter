#include "global.h"	// include first in any cpp
#include <glm/ext/matrix_transform.inl>
#include "RenderComponent.h"
#include "vulkanBase.h"
#include "Renderer.h"
#include "Material.h"


CObjManager<CRenderCompHandle, RenderComponent> g_MeshManager;

// todo
void Log(const char* format, ...);

RenderComponent::RenderComponent() : m_vertices(EF_Type_Buffer)
	, m_indices(EF_Type_Buffer)
	, m_BLAS(EF_Type_Buffer)
	, m_primitiveMaterials(EF_Type_Buffer)
{
}

RenderComponent::RenderComponent(const char* inFilePath, const float meshScale)
	: m_filePath(inFilePath)
	, m_meshScale(meshScale)
	, m_vertices(EF_Type_Buffer)
	, m_indices(EF_Type_Buffer)
	, m_BLAS(EF_Type_Buffer)
	, m_primitiveMaterials(EF_Type_Buffer)
{
	ASSERT(inFilePath);
}

void RenderComponent::deinit(EventContext& context)
{
	if (m_BottomLevelAS != VK_NULL_HANDLE)
	{
		vkDestroyAccelerationStructureKHR(context.device, m_BottomLevelAS, nullptr);
		m_BottomLevelAS = VK_NULL_HANDLE;
	}
	m_indices.deinit(context);
	m_vertices.deinit(context);
	m_primitiveMaterials.deinit(context);
	m_BLAS.deinit(context);
}

void RenderComponent::createFromSource(VulkanContext& vulkanContext)
{
	if(m_sourceIndices.empty())
		return;

	m_indices.initRaw(vulkanContext.allocator, vulkanContext.cmd, m_sourceIndices.data(), m_sourceIndices.size(), sizeof(*m_sourceIndices.data()), Buffer::eBT_IndexBuffer);
	m_vertices.initRaw(vulkanContext.allocator, vulkanContext.cmd, m_sourceVertices.data(), m_sourceVertices.size(), sizeof(*m_sourceVertices.data()), Buffer::eBT_VertexBuffer);
}

void RenderComponent::buildPrimitiveMaterials(VulkanContext& vulkanContext, uint32 triangleCount)
{
	if (triangleCount == 0)
		return;

	std::vector<uint32> primitiveMaterials(triangleCount, shaderio::kInvalidMaterialIndex);

	for (const SubMesh& sub : m_subMeshes)
	{
		ASSERT(sub.hndMaterial.IsValid());  // never optional -- see rayClosestHitMain, ray.slang
		// rayClosestHitMain indexes g_materialBuffer (rebuilt every frame from
		// g_MaterialManager.GetElementByIterator(), see App::dispatchRayTracing) directly
		// by this raw Index, not through GetElementPtr() -- an out-of-range value here is a GPU
		// out-of-bounds read (silent corruption or VK_ERROR_DEVICE_LOST) instead of a clean assert.
		ASSERT(sub.hndMaterial.Index < g_MaterialManager.GetElementCount());
		const uint32 triStart = sub.indexStart / 3;
		const uint32 triCount = sub.indexCount / 3;
		for (uint32 t = 0; t < triCount; ++t)
			primitiveMaterials[triStart + t] = sub.hndMaterial.Index;
	}

	// Every triangle must be covered by some submesh -- a leftover kInvalidMaterialIndex here means
	// m_subMeshes has a gap, which would read out of bounds of the material buffer at trace time.
	for (uint32 matIndex : primitiveMaterials)
		ASSERT(matIndex != shaderio::kInvalidMaterialIndex);

	m_primitiveMaterials.initRaw(vulkanContext.allocator, vulkanContext.cmd, primitiveMaterials.data(),
		primitiveMaterials.size(), sizeof(uint32), Buffer::eBT_ShaderBindingTable);
}

static CMatHandle getFallbackMaterial()
{
	Material m;
	m.m_albedoTextureFade = 0.0f;  // no per-material texture to fade in
	return g_MaterialManager.push_back(m);
}

void RenderComponent::rebuildSingleSubMesh(VulkanContext& vulkanContext)
{
	m_subMeshes.clear();

	SubMesh sub;
	sub.indexStart = 0;
	sub.indexCount = (uint32)m_sourceIndices.size();
	// m_proceduralMaterial round-trips through save/load (see saveMeshes()/AppendArena.cpp) --
	// only falls back to the generic placeholder if this mesh was never given one (e.g. a
	// corrupt/hand-edited scene.aa), not on every reload.
	sub.hndMaterial = m_proceduralMaterial.IsValid() ? m_proceduralMaterial : getFallbackMaterial();
	m_subMeshes.push_back(sub);

	buildPrimitiveMaterials(vulkanContext, sub.indexCount / 3);
}

void RenderComponent::createPlane(VulkanContext& vulkanContext, float halfSize, CMatHandle material)
{
	ASSERT(material.IsValid());
	m_proceduralMaterial = material;

	std::vector<uint32> indices = { 0, 2, 1, 2, 0, 3 };
	std::vector<shaderio::Vertex> vertices;
	vertices.resize(6, shaderio::Vertex{});

	vertices[0].position = float3(-halfSize, 0, -halfSize);
	vertices[1].position = float3(halfSize, 0, -halfSize);
	vertices[2].position = float3(halfSize, 0, halfSize);
	vertices[3].position = float3(-halfSize, 0, halfSize);

	vertices[0].texCoord = float2(0, 0);
	vertices[1].texCoord = float2(1, 0);
	vertices[2].texCoord = float2(1, 1);
	vertices[3].texCoord = float2(0, 1);

	for (int i = 0; i < 4; ++i)
	{
		vertices[i].normal = float3(0, 1, 0);
		// texCoord.x increases along +X (see the texCoord assignments above), so tangent = +X;
		// texCoord.y increases along +Z, giving bitangent = +Z, but cross(normal, tangent) =
		// cross((0,1,0),(1,0,0)) = (0,0,-1) -- the opposite direction -- hence handedness -1 (see
		// ray.slang's TBN construction, "B = cross(N, T) * tangent.w", for where this sign is used).
		vertices[i].tangent = float4(1, 0, 0, -1);
	}

	std::swap(indices, m_sourceIndices);
	std::swap(vertices, m_sourceVertices);
	createFromSource(vulkanContext);

	rebuildSingleSubMesh(vulkanContext);
}