#include "global.h"	// include first in any cpp
#include "tinygltf/tiny_gltf_v3.h"
#include "tg3_gltf_loader.h"
#include <glm/ext/matrix_transform.inl>
#include "RenderComponent.h"
#include "vulkanBase.h"
#include "Renderer.h"
#include "Material.h"
#include "AppendArena.h"


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

int findAttribute(const tg3_primitive& primitive, const char* name)
{
	for (uint32 i = 0; i < primitive.attributes_count; ++i)
	{
		const auto& a = primitive.attributes[i];

		if (strcmp(a.key.data, name) == 0)
			return a.value;
	}

	return -1;
}

template<class T>
const T* getAccessorData(const tg3_model& model, int accessorIndex)
{
	const tg3_accessor& accessor = model.accessors[accessorIndex];
	const tg3_buffer_view& view = model.buffer_views[accessor.buffer_view];
	const tg3_buffer& buffer = model.buffers[view.buffer];

	const uint8* ptr =
		buffer.data.data +
		view.byte_offset +
		accessor.byte_offset;

	return reinterpret_cast<const T*>(ptr);
}

#if USE_RAY_TRACING
static void buildBLAS(VulkanContext& vulkanContext, Buffer& indices, Buffer& vertices, Buffer& blas, VkAccelerationStructureKHR& bottomLevelAS)
{
	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	uint32 primitiveCount = (uint32)(indices.m_size / indices.m_stride / 3);

	VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };

	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	geometry.geometry.triangles.vertexData.deviceAddress = vertices.m_data.address;
	geometry.geometry.triangles.vertexStride = (uint32)sizeof(Vertex);
	geometry.geometry.triangles.maxVertex = (uint32)(vertices.m_size / vertices.m_stride);
	geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
	geometry.geometry.triangles.indexData.deviceAddress = indices.m_data.address;
	geometry.geometry.triangles.transformData.deviceAddress = 0;

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };

	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &geometry;

	vkGetAccelerationStructureBuildSizesKHR(
		vulkanContext.device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&buildInfo,
		&primitiveCount,
		&sizeInfo);

	// A mesh can be (re)built more than once now (e.g. "setup #1" clicked again) -- destroy the
	// previous BLAS handle first, or it leaks (caught by the validation layer at vkDestroyDevice,
	// same as every other such leak fixed this session). blas.init() below already defers freeing
	// the old buffer safely (Buffer::init()).
	if (bottomLevelAS != VK_NULL_HANDLE)
	{
		vkDestroyAccelerationStructureKHR(vulkanContext.device, bottomLevelAS, nullptr);
		bottomLevelAS = VK_NULL_HANDLE;
	}

	blas.init(vulkanContext.allocator, sizeInfo.accelerationStructureSize,
		VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);

	VkAccelerationStructureCreateInfoKHR asCreateInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
	asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	asCreateInfo.buffer = blas.m_data.buffer;
	asCreateInfo.offset = 0;
	asCreateInfo.size = sizeInfo.accelerationStructureSize;

	VK_CHECK(vkCreateAccelerationStructureKHR(
		vulkanContext.device,
		&asCreateInfo,
		nullptr,
		&bottomLevelAS));

	// vkCmdBuildAccelerationStructuresKHR requires buildInfo.scratchData.deviceAddress to be a
	// multiple of minAccelerationStructureScratchOffsetAlignment -- same requirement as the TLAS
	// build in Renderer.cpp, passed straight through to Buffer::init()'s minAlignment.
	VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
	{
		VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
		props2.pNext = &asProps;
		vkGetPhysicalDeviceProperties2(vulkanContext.physicalDevice, &props2);
	}

	Buffer scratchMemory(EF_Type_Unknown);

	scratchMemory.init(vulkanContext.allocator, sizeInfo.buildScratchSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
		asProps.minAccelerationStructureScratchOffsetAlignment);

	// Filled in later
	buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
	buildInfo.dstAccelerationStructure = bottomLevelAS;
	buildInfo.scratchData.deviceAddress = scratchMemory.m_data.address;

	VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};

	rangeInfo.primitiveCount = primitiveCount;
	rangeInfo.primitiveOffset = 0;
	rangeInfo.firstVertex = 0;
	rangeInfo.transformOffset = 0;
	const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

	vkCmdBuildAccelerationStructuresKHR(vulkanContext.cmd, 1, &buildInfo, &pRange);

	VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };

	barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
	barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

	vkCmdPipelineBarrier(vulkanContext.cmd,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		0, 1, &barrier, 0, nullptr, 0, nullptr);

	EventContext context { vulkanContext.device, vulkanContext.allocator };
	// vulkanContext.cmd is the caller's shared, batched command buffer (see Mesh::create/load's
	// callers in App.cpp): it's only *recorded* into here, not submitted until much
	// later, after every mesh, image, and the descriptor heap have all been recorded into it too.
	// deinit() is safe to call right away regardless -- it defers the actual free until the GPU
	// has proven it's done with this buffer (DeferredRelease.h), rather than freeing immediately
	// like the old destroyBuffer() call here used to (a confirmed, reproduced use-after-free /
	// VK_ERROR_DEVICE_LOST, root cause of a session-long crash hunt).
	scratchMemory.deinit(context);
}
#endif

void RenderComponent::createFromSource(VulkanContext& vulkanContext)
{
	if(m_sourceIndices.empty())
		return;

	m_indices.initRaw(vulkanContext.allocator, vulkanContext.cmd, m_sourceIndices.data(), m_sourceIndices.size(), sizeof(*m_sourceIndices.data()), Buffer::eBT_IndexBuffer);
	m_vertices.initRaw(vulkanContext.allocator, vulkanContext.cmd, m_sourceVertices.data(), m_sourceVertices.size(), sizeof(*m_sourceVertices.data()), Buffer::eBT_VertexBuffer);

#if USE_RAY_TRACING
	buildBLAS(vulkanContext, m_indices, m_vertices, m_BLAS, m_BottomLevelAS);
#endif
}

// Fallback for a glTF primitive with no material (glTF's prim.material == -1): ray tracing's
// per-triangle material lookup (m_primitiveMaterials) is never optional, so every submesh needs a
// real handle, even ones the source file never assigned one to. Flat mid-gray, no texture.
// Not cached across calls: a `static CMatHandle` here used to memoize the handle for the life of
// the process, but g_MaterialManager itself gets fully swapped out for a fresh one on every scene
// reload (see loadScene(), App.cpp) -- the cached handle's Index then refers to
// whatever (possibly nonexistent, possibly wrong) slot happened to be there in the new manager,
// an out-of-bounds/wrong-material read in ray.slang's rayClosestHitMain. Creating a fresh one each
// call is simplest and correct; the few extra tiny Material entries this can add are a non-issue.
static CMatHandle getFallbackMaterial()
{
	Material m;
	m.m_albedoTextureFade = 0.0f;  // no per-material texture to fade in
	return g_MaterialManager.push_back(m);
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

void RenderComponent::load(VulkanContext& vulkanContext, const glm::mat4& extraTransform)
{
	m_loadFromFile = true;

	tg3_parse_options opts;
	tg3_error_stack errors;
	tg3_model model;

	tg3_parse_options_init(&opts);
	tg3_error_stack_init(&errors);

	tg3_error_code err = tg3_parse_file(&model, &errors, m_filePath.c_str(), (uint32)m_filePath.size(), &opts);
	if (err != TG3_OK)
	{
		for (uint32 i = 0; i < errors.count; i++)
		{
			Log("[%d] %s\n", (int)errors.entries[i].severity, errors.entries[i].message ? errors.entries[i].message : "(null)");
		}
	}

	{
		std::vector<shaderio::Vertex> vertices;

		GltfMeshData meshData;

		tg3_load_meshes(model, meshData);
		tg3_load_materials(model, vulkanContext, m_materialHandles);

		// axis for debugging
//		if(1)
//			meshData.addAxis(float3(0, 0, 0), float3(10, 20, 10), 0.1f);

		const glm::mat4& transform = glm::scale(extraTransform, float3(m_meshScale, m_meshScale, m_meshScale));

		// Normals need the inverse-transpose of the transform, not the transform itself, to stay
		// correct under non-uniform scale (same reasoning as tg3_gltf_loader.cpp's per-node
		// normalMatrix) -- this was previously a no-op (vectorTransform was computed but never
		// used below; harmless only because the old hardcoded transform was a pure rotation, its
		// own inverse-transpose). Now that callers can pass an arbitrary matrix here, it has to
		// actually be used. Tangents transform with the matrix directly (standard convention:
		// only normals need the correction).
		const glm::mat4 normalTransform = glm::transpose(glm::inverse(transform));

		vertices.resize(meshData.vertices.size());
		auto* dst = vertices.data();
		for (auto it = meshData.vertices.begin(), end = meshData.vertices.end(); it != end; ++it)
		{
			auto src = *it;
			dst->position = float3(transform * float4(src.position, 1.0f));
			dst->texCoord = src.uv;
			dst->normal = float3(normalTransform * float4(src.normal, 0.0f));
			dst->tangent = float4(float3(transform * float4(float3(src.tangent), 0.0f)), src.tangent.w);
			++dst;
		}

		// create() calls initRaw() on m_indices/m_vertices itself (and builds the BLAS from them) --
		// calling initRaw() again here would leak the buffers just created above, since the second
		// call overwrites Buffer::m_data before the first allocation is ever freed.
		std::swap(meshData.indices, m_sourceIndices);
		std::swap(vertices, m_sourceVertices);
		createFromSource(vulkanContext);

		m_subMeshes.reserve(meshData.subMeshes.size());
		for (auto it = meshData.subMeshes.begin(), end = meshData.subMeshes.end(); it != end; ++it)
		{
			SubMesh sub;
			sub.indexStart = it->indexStart;
			sub.indexCount = it->indexCount;
			if (it->materialIndex >= 0 && (size_t)it->materialIndex < m_materialHandles.size())
				sub.hndMaterial = m_materialHandles[it->materialIndex];
			else
				sub.hndMaterial = getFallbackMaterial();  // glTF primitive had no material -- never optional, see m_primitiveMaterials
			m_subMeshes.push_back(sub);
		}

		// Per-triangle material lookup for the ray tracing path (see m_primitiveMaterials'
		// declaration comment) -- built from the same m_subMeshes ranges the rasterizer's draw loop
		// already uses, so both paths render each glTF primitive with the same material.
		//
		// meshData.indices was swapped into m_sourceIndices above (now empty) -- read the triangle
		// count from its new home, not the emptied local, or this silently computes 0 triangles and
		// buildPrimitiveMaterials() early-returns without ever building m_primitiveMaterials. The
		// ray tracing shader then reads a null per-triangle-material buffer address (rayClosestHitMain,
		// ray.slang), an out-of-bounds GPU read that manifests as VK_ERROR_DEVICE_LOST.
		buildPrimitiveMaterials(vulkanContext, (uint32)(m_sourceIndices.size() / 3));
	}

	tg3_model_free(&model);
	tg3_error_stack_free(&errors);
}

void saveMeshes(TokenAppendArena& arena)
{
	auto& container = g_MeshManager;

	for (uint32 i = 0, count = container.GetElementCount(); i < count; ++i)
	{
		// todo: could be optimized to single function call
		auto hnd = container.GetHandleByIterator(i);
		auto& el = container.GetElementByIterator(i);

		TypedHandle dstKey(hnd, 3);	// 3 is Mesh

		// A procedural mesh (e.g. g_plane, built via createPlane()) has no file to reload from --
		// m_filePath can still be non-empty (a debug label, not a real path), so this flag (set by
		// load(), never by createPlane()) is what distinguishes them. See its declaration in Mesh.h.
//		if (!el.m_loadFromFile)
//			continue;

		LOGI_AA_SAVE("Mesh[%d] name=%s", dstKey.Index, el.m_filePath.c_str());

		// todo: use reflection
		{
			ATokenCmdNew cmd;
			cmd.dstKey = dstKey;
			arena.appendCmd(cmd);
		}
		{
			// todo: improve API !
			ATokenCmdBlob blob(el.m_filePath.size() + 1);
			arena.appendCmd(blob);

			size_t oldSize = arena.size();
			uint64 value0 = arena.appendString(el.m_filePath);
			size_t newSize = arena.size();
			ASSERT(newSize == oldSize + blob.count);

			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 0;
			cmd.valueOrIndexInArenaToValue = value0;
			arena.appendCmd(cmd);
		}

		// todo: m_materialHandles

		if(!el.m_loadFromFile)	// save IB and VB
		{
			// todo: make a save cmd/function for std::vectors
			// getVector() (AppendArena.cpp) reads this length prefix as an *element* count
			// (out.resize(count)), so that's what must be written here -- not a byte count.
			uint32 countIB = (uint32)el.m_sourceIndices.size();
			uint32 countVB = (uint32)el.m_sourceVertices.size();
			uint32 sizeIB = countIB * (uint32)sizeof(el.m_sourceIndices[0]);
			uint32 sizeVB = countVB * (uint32)sizeof(el.m_sourceVertices[0]);

			// todo: improve API !
			ATokenCmdBlob blob(sizeIB + sizeVB + sizeof(uint32) * 2);
			arena.appendCmd(blob);

			size_t oldSize = arena.size();

			// appendBytes() takes a byte count -- passing 1 instead of sizeof(count*) wrote only
			// the prefix's first byte, leaving getVector()'s uint32 read to pick up 3 zeroed bytes
			// from the rest of that (otherwise-unwritten) token alongside it.
			uint64 valueIB = arena.appendBytes(&countIB, sizeof(countIB));
			arena.appendBytes(el.m_sourceIndices.data(), sizeIB);

			uint64 valueVB = arena.appendBytes(&countVB, sizeof(countVB));
			arena.appendBytes(el.m_sourceVertices.data(), sizeVB);

			size_t newSize = arena.size();
			ASSERT(newSize == oldSize + blob.count);

			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 1;
			cmd.valueOrIndexInArenaToValue = valueIB;
			arena.appendCmd(cmd);
			cmd.memberIndex = 2;
			cmd.valueOrIndexInArenaToValue = valueVB;
			arena.appendCmd(cmd);
			cmd.memberIndex = 3;
			cmd.valueOrIndexInArenaToValue = el.m_proceduralMaterial.getRaw();
			arena.appendCmd(cmd);
		}

		// Full per-submesh material assignment (indexStart/indexCount/hndMaterial for every entry
		// in m_subMeshes), not just the single m_proceduralMaterial above -- needed for any
		// procedural mesh with more than one submesh, e.g. a loaded BSP level (see
		// 3DBSPLevel.cpp: one submesh per unique lightmap/texture combo, referencing many different
		// materials). Without this, loadScene()'s reload used to always call rebuildSingleSubMesh()
		// and collapse the whole mesh onto m_proceduralMaterial (invalid for a BSP level, which
		// never sets it), losing every face's real texture/lightmap -- confirmed live as "setup #2"
		// -> save -> load losing all BSP textures. Saved/restored regardless of submesh count (a
		// single-submesh procedural mesh like g_plane just gets a length-1 array here), so
		// loadScene() no longer needs rebuildSingleSubMesh() at all except as a defensive fallback
		// for an old save file with no memberIndex 4 data.
		if (!el.m_loadFromFile)
		{
			uint32 subMeshCount = (uint32)el.m_subMeshes.size();

			struct SavedSubMesh { uint32 indexStart; uint32 indexCount; uint64 hndMaterialRaw; };
			std::vector<SavedSubMesh> saved(subMeshCount);
			for (uint32 s = 0; s < subMeshCount; ++s)
				saved[s] = { el.m_subMeshes[s].indexStart, el.m_subMeshes[s].indexCount, el.m_subMeshes[s].hndMaterial.getRaw() };

			uint32 sizeSubMeshes = subMeshCount * (uint32)sizeof(SavedSubMesh);

			ATokenCmdBlob blob(sizeof(uint32) + sizeSubMeshes);
			arena.appendCmd(blob);

			size_t oldSize = arena.size();
			uint64 valueSubMeshes = arena.appendBytes(&subMeshCount, sizeof(subMeshCount));
			// appendBytes() asserts sizeInBytes != 0 -- a mesh with zero submeshes (e.g. a test's
			// minimal procedural mesh, which never populates m_subMeshes at all) would otherwise
			// assert here (confirmed live as SceneTests.cpp hanging on a debug-assert dialog).
			if (sizeSubMeshes)
				arena.appendBytes(saved.data(), sizeSubMeshes);
			size_t newSize = arena.size();
			ASSERT(newSize == oldSize + blob.count);

			ATokenCmdSetMember cmd2;
			cmd2.dstKey = dstKey;
			cmd2.memberIndex = 4;
			cmd2.valueOrIndexInArenaToValue = valueSubMeshes;
			arena.appendCmd(cmd2);
		}
	}
}

// Like getVector() (AppendArena.h), but for m_subMeshes specifically: each entry's hndMaterial is
// a handle and needs remapping (see getHandle()'s own comment) -- unlike getVector()'s raw memcpy,
// which would leave every submesh pointing at its *old*, as-saved material index instead of
// wherever that material actually landed this load.
static void getSubMeshes(ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping, std::vector<RenderComponent::SubMesh>& out)
{
	const AToken* p = inoutRange.begin() + cmd->valueOrIndexInArenaToValue;
	uint32 count = *(uint32*)p;

	struct SavedSubMesh { uint32 indexStart; uint32 indexCount; uint64 hndMaterialRaw; };
	const SavedSubMesh* saved = (const SavedSubMesh*)(p + 1);

	out.resize(count);
	for (uint32 i = 0; i < count; ++i)
	{
		out[i].indexStart = saved[i].indexStart;
		out[i].indexCount = saved[i].indexCount;

		CObjHandle oldHnd;
		oldHnd.setRaw(saved[i].hndMaterialRaw);
		if (oldHnd.IsValid())
		{
			CObjHandle newHnd = remapping.Get(1, oldHnd.Index);	// 1 = Material
			out[i].hndMaterial.Seed = newHnd.Seed;
			out[i].hndMaterial.Index = newHnd.Index;
		}
	}
	LOGI_AA_LOAD("set [%d][%d] = %u submeshes", cmd->dstKey.Index, cmd->memberIndex, count);
}

void load(RenderComponent& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping)
{
	// todo: reflection with backward compatible loading
	if (cmd->memberIndex == 0) get(cmd, inoutRange, ref.m_filePath);
	else if (cmd->memberIndex == 1) getVector(cmd, inoutRange, ref.m_sourceIndices);
	else if (cmd->memberIndex == 2) getVector(cmd, inoutRange, ref.m_sourceVertices);
	else if (cmd->memberIndex == 3) getHandle(cmd, remapping, 1, ref.m_proceduralMaterial);
	else if (cmd->memberIndex == 4) getSubMeshes(cmd, inoutRange, remapping, ref.m_subMeshes);
	else ASSERT(0);
}

