#pragma once
#include "Buffer.h"
#include <string>
#include "ObjManager.h"
#include <glm/glm.hpp>

struct VulkanContext;
class TokenAppendArena;
struct ATokenCmdSetMember;
struct ArenaTokenRange;
class HandleRemapTable;

class RenderComponent
{
	float m_meshScale = 1.0f;
	// for ray tracing
	Buffer m_BLAS;


public: // todo
	// [0] e.g. "resources/SheenWoodLeatherSofa.glb"
	std::string m_filePath;
	// true only if load() (below) actually built this mesh from m_filePath -- false for a
	// procedural mesh (e.g. createPlane()), which has no file to reload from. m_filePath can still
	// be set (and non-empty) for those -- e.g. g_plane's is "plane", a debug label, not a real path
	// -- so this flag, not an empty-string check, is what saveMeshes() must gate serialization on:
	// reloading a procedural mesh via load() would call tg3_parse_file() on a nonexistent path,
	// leaving it with no geometry, and Buffer::initRaw() asserts on the resulting null data.
	bool m_loadFromFile = false;

public:
	RenderComponent();
	// constructor
	// @param inFilePath name or filepath e.g. "resources/test.glb", must not be 0
	RenderComponent(const char* inFilePath, const float meshScale = 1.0f);

	// Y is up, around 0,0,0. material must be valid -- every mesh always carries a material now,
	// see m_primitiveMaterials.
	void createPlane(VulkanContext& vulkanContext, float halfSize, CMatHandle material);

	// Releases every GPU resource this mesh owns (m_indices/m_vertices/m_primitiveMaterials/m_BLAS,
	// plus the plain VkAccelerationStructureKHR m_BottomLevelAS, which isn't a Buffer so deinit()
	// alone never touches it) -- call before this MeshComponent's own destructor can run, or
	// Buffer::~Buffer()'s "forgot to call deinit()" assert fires for each of the first four (see
	// e.g. loadScene()'s g_MeshManager reset, App.cpp, which -- now that g_MeshManager
	// stores MeshComponent by value instead of MeshComponent* -- actually destructs every mesh it's
	// discarding, not just stops tracking them).
	void deinit(EventContext& context);

	//
	// Default-initialized here (not left to the constructors, neither of which touches it) since
	// buildBLAS() (MeshComponent.cpp) destroys whatever this holds before rebuilding it, checking
	// only "!= VK_NULL_HANDLE" -- harmless for a global MeshComponent (static storage duration
	// zero-inits it anyway, same reasoning as Camera.h's absRot), but a MeshComponent loaded from a
	// saved scene is heap-allocated via `new` (see ATokenCmdNew::execute(), AppendArena.cpp), which
	// leaves this as uninitialized garbage -- almost never VK_NULL_HANDLE, so buildBLAS() would call
	// vkDestroyAccelerationStructureKHR() on a handle that was never valid (observed as a
	// non-deterministic VK_ERROR_DEVICE_LOST after a handful of repeated "load" clicks).
	VkAccelerationStructureKHR m_BottomLevelAS = VK_NULL_HANDLE;
	// vertex buffer
	Buffer m_vertices;
	// index buffer
	Buffer m_indices;
	// one handle per model.materials[] entry from the loaded glTF file (index-aligned), empty for
	// createPlane() or if the file had no materials. See tg3_load_materials (tg3_gltf_loader.h) --
	// assigning these to scene objects/primitives is not done here, do that manually.
	std::vector<CMatHandle> m_materialHandles;

	// [1] todo: consider moving out, for now needed for serialization of objects not loaded
	std::vector<uint32> m_sourceIndices;
	// [2]
	std::vector<shaderio::Vertex> m_sourceVertices;
	// [3] The single material a procedural mesh (createPlane()) uses for its whole surface --
	// invalid (IsValid()==false) for a file-loaded mesh, which gets its per-primitive materials
	// from the glTF file instead (see m_materialHandles/load()). Saved/restored alongside
	// m_sourceIndices/m_sourceVertices (see saveMeshes()) so rebuildSingleSubMesh() can recreate
	// the same submesh->material association after a reload instead of falling back to a generic
	// placeholder material.
	CMatHandle m_proceduralMaterial;

	// One glTF primitive's worth of the combined m_indices buffer (still one shared vertex/index
	// Buffer per Mesh, no per-primitive GPU buffers) plus the material it should draw with.
	struct SubMesh
	{
		// range into m_indices, in index units (not bytes)
		uint32 indexStart = 0;
		uint32 indexCount = 0;
		// the glTF primitive's material; invalid (IsValid()==false) if it had none
		CMatHandle hndMaterial;
	};
	// one entry per glTF primitive across every node in the loaded file (see tg3_load_meshes'
	// GltfMeshData::subMeshes), empty for createPlane().
	std::vector<SubMesh> m_subMeshes;

	// One uint32 per triangle in m_indices (m_indices.count/3 entries): the resolved
	// CObjHandle::Index of that triangle's m_subMeshes material -- always a valid index, never
	// shaderio::kInvalidMaterialIndex (see buildPrimitiveMaterials). The ray tracing path can only
	// carry one material per TLAS instance (instanceCustomIndex), unlike the rasterizer's
	// per-submesh draw loop -- rayClosestHitMain (ray.slang) reads this buffer via PrimitiveIndex()
	// to get the same per-primitive material the rasterizer already uses. Every mesh has this built
	// (see load()/createPlane()/3DBSPLevel::Load()), so the address is never 0 by the time a ray
	// tracing dispatch can reach it.
	Buffer m_primitiveMaterials;

	// Builds m_primitiveMaterials from m_subMeshes: every triangle in [0, triangleCount) must be
	// covered by exactly one submesh with a valid material -- ray tracing's per-triangle material
	// lookup is not optional (see rayClosestHitMain, ray.slang). Call once m_subMeshes is fully
	// populated and the index buffer exists (i.e. after create()/initRaw()).
	void buildPrimitiveMaterials(VulkanContext& vulkanContext, uint32 triangleCount);

//private:
//	void create(VulkanContext& vulkanContext, std::vector<uint32>& indices, std::vector<shaderio::Vertex>& vertices);
	// input is m_sourceIndices and m_sourceVertices
	void createFromSource(VulkanContext& vulkanContext);

	// Builds the single submesh a procedural mesh (createPlane(), or a reload of one -- see
	// createFromSource()) needs: one submesh spanning the whole mesh, using m_proceduralMaterial
	// (round-tripped through save/load) if valid, otherwise the same fallback material a glTF
	// primitive with no material gets (see getFallbackMaterial(), MeshComponent.cpp).
	// m_primitiveMaterials is never optional -- ray tracing's per-triangle material lookup
	// (rayClosestHitMain, ray.slang) needs it, or the shader reads a null buffer address
	// (VK_ERROR_DEVICE_LOST). Call once m_sourceIndices/createFromSource() have run.
	void rebuildSingleSubMesh(VulkanContext& vulkanContext);
};

extern CObjManager<CRenderCompHandle, RenderComponent> g_MeshManager;