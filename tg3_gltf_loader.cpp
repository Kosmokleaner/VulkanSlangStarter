#include "global.h"	// include first in any cpp
#include "tg3_gltf_loader.h"
#include "Material.h"
#include "Texture.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

// glTF node local transform: node.matrix directly if present (glTF matrices are column-major,
// same layout glm::mat4's column-major constructor expects), otherwise composed from the
// translation/rotation/scale fields. glTF stores quaternions as (x,y,z,w); glm::quat's
// constructor takes (w,x,y,z) -- easy to get backwards, so called out explicitly here.
static glm::mat4 tg3_node_local_matrix(const tg3_node& node)
{
	if (node.has_matrix)
	{
		const double* m = node.matrix;
		return glm::mat4(
			(float)m[0], (float)m[1], (float)m[2], (float)m[3],
			(float)m[4], (float)m[5], (float)m[6], (float)m[7],
			(float)m[8], (float)m[9], (float)m[10], (float)m[11],
			(float)m[12], (float)m[13], (float)m[14], (float)m[15]);
	}

	const float3 t((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]);
	const glm::quat r((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]);
	const float3 s((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]);

	return glm::translate(glm::mat4(1.0f), t) * glm::mat4(r) * glm::scale(glm::mat4(1.0f), s);
}

// Recurses into node.children, accumulating each node's local transform into parentWorld, and
// appends node.mesh's geometry (if any) transformed into world space. A node with no mesh is
// just a transform pivot for its children (e.g. an empty "group" node) -- still walked for its
// children, just contributes no vertices itself.
static void tg3_load_node(const tg3_model& model, int32_t nodeIndex, const glm::mat4& parentWorld, GltfMeshData& out)
{
	const tg3_node& node = model.nodes[nodeIndex];
	const glm::mat4 world = parentWorld * tg3_node_local_matrix(node);

	if (node.mesh >= 0)
	{
		const tg3_mesh& mesh = model.meshes[node.mesh];
		// Normals (and tangents' handedness) need the inverse-transpose under non-uniform scale,
		// not the transform itself -- same reasoning as Mesh.cpp's rayClosestHitMain comment for
		// WorldToObject3x4(). Tangents use `world` directly (standard convention: tangents
		// transform with the linear part of the model matrix, only normals need the correction).
		const glm::mat4 normalMatrix = glm::transpose(glm::inverse(world));

		for (uint32_t pi = 0; pi < mesh.primitives_count; pi++)
		{
			const tg3_primitive& prim = mesh.primitives[pi];

			int pos = tg3_find_attribute(prim, "POSITION");

			if (pos < 0)
				continue;

			int normal = tg3_find_attribute(prim, "NORMAL");
			int uv = tg3_find_attribute(prim, "TEXCOORD_0");
			int tangent = tg3_find_attribute(prim, "TANGENT");

			uint32_t base = (uint32_t)out.vertices.size();

			const tg3_accessor& posAcc = model.accessors[pos];

			for (uint32_t i = 0; i < posAcc.count; i++)
			{
				GLTFVertex v{};

				v.position = float3(world * float4(tg3_read<float3>(model, pos, i), 1.0f));

				if (normal >= 0)
					v.normal = float3(normalMatrix * float4(tg3_read<float3>(model, normal, i), 0.0f));

				if (uv >= 0)
					v.uv = tg3_read<float2>(model, uv, i);

				if (tangent >= 0)
				{
					const float4 t = tg3_read<float4>(model, tangent, i);
					const float3 tXformed = float3(world * float4(float3(t), 0.0f));
					v.tangent = float4(tXformed, t.w);
				}

				out.vertices.push_back(v);
			}


			const uint32_t indexStart = (uint32_t)out.indices.size();

			if (prim.indices >= 0)
			{
				const tg3_accessor& ia = model.accessors[prim.indices];

				for (uint32_t i = 0; i < ia.count; i++)
					out.indices.push_back(base + tg3_read_index(model, ia, i));
			}
			else
			{
				for (uint32_t i = 0; i < posAcc.count; i++)
					out.indices.push_back(base + i);
			}

			GltfSubMeshRange sub;
			sub.indexStart = indexStart;
			sub.indexCount = (uint32_t)out.indices.size() - indexStart;
			sub.materialIndex = prim.material;
			out.subMeshes.push_back(sub);
		}
	}

	for (uint32_t ci = 0; ci < node.children_count; ci++)
		tg3_load_node(model, node.children[ci], world, out);
}

// very inefficient
//
// Walks the glTF scene graph (the default scene's root nodes, recursing into children) instead
// of dumping model.meshes directly: model.meshes is just geometry data with no placement --
// node.mesh + the node's own transform (and its ancestors') is what actually says "this geometry
// goes here, at this position/rotation/scale". Reading model.meshes directly, as this used to,
// silently applied no per-object transform at all (every mesh landed at its raw local-space
// coordinates) and would draw a mesh shared by multiple nodes (instancing) only once.
bool tg3_load_meshes(const tg3_model& model, GltfMeshData& out)
{
	if (model.scenes_count > 0)
	{
		const int32_t sceneIndex = model.default_scene >= 0 ? model.default_scene : 0;
		const tg3_scene& scene = model.scenes[sceneIndex];

		for (uint32_t i = 0; i < scene.nodes_count; i++)
			tg3_load_node(model, scene.nodes[i], glm::mat4(1.0f), out);
	}
	else
	{
		// No scene defined (unusual/malformed file) -- fall back to walking every node directly
		// rather than silently loading nothing. May double-process a node reachable both as a
		// "root" here and as another node's child, but that's better than an empty mesh.
		for (uint32_t i = 0; i < model.nodes_count; i++)
			tg3_load_node(model, (int32_t)i, glm::mat4(1.0f), out);
	}

	return true;
}

// Loads model.textures[textureIndex]'s image and registers it as a Texture (see Texture.h),
// returning its handle -- invalid (IsValid()==false) if textureIndex is absent (-1) or the image
// isn't glb-embedded (referenced via a bufferView into the binary chunk, the common case for a
// single-file .glb); external file/data-uri images are not yet supported and are skipped with a
// log line instead.
static CTexHandle tg3_load_embedded_texture(const tg3_model& model, VulkanContext& vulkanContext, int32_t textureIndex, const std::string& debugName, bool isSRGB)
{
	if (textureIndex < 0 || (uint32_t)textureIndex >= model.textures_count)
		return CTexHandle();

	const tg3_texture& tex = model.textures[textureIndex];

	if (tex.source < 0 || (uint32_t)tex.source >= model.images_count)
		return CTexHandle();

	const tg3_image& img = model.images[tex.source];

	if (img.buffer_view < 0)
	{
		LOGI("glTF texture \"%s\" isn't glb-embedded (no bufferView) -- skipped\n", debugName.c_str());
		return CTexHandle();
	}

	const tg3_buffer_view& bv = model.buffer_views[img.buffer_view];
	const tg3_buffer& buf = model.buffers[bv.buffer];
	const uint8_t* bytes = buf.data.data + bv.byte_offset;

	Texture texture;
	// See Texture.h's m_isSRGB comment: base-color is gamma-encoded color data (GPU decodes it for
	// free on sample), metallic-roughness is linear non-color data and must not be decoded.
	texture.m_isSRGB = isSRGB;
	texture.loadFromMemory(vulkanContext, bytes, (size_t)bv.byte_length, debugName);

	extern CObjManager<CTexHandle, Texture> g_TextureManager;
	return g_TextureManager.push_back(texture);
}

bool tg3_load_materials(const tg3_model& model, VulkanContext& vulkanContext, std::vector<CMatHandle>& outMaterialHandles)
{
	extern CObjManager<CMatHandle, Material> g_MaterialManager;

	outMaterialHandles.clear();
	outMaterialHandles.reserve(model.materials_count);

	for (uint32_t i = 0; i < model.materials_count; i++)
	{
		const tg3_material& gltfMat = model.materials[i];
		const tg3_pbr_metallic_roughness& pbr = gltfMat.pbr_metallic_roughness;

		Material mat;

		// can be optimized
		mat.m_name = std::string(gltfMat.name.data, gltfMat.name.len);
		const std::string debugName = mat.m_name.empty() ? "gltf material texture" : mat.m_name;
		// Only what Material currently supports. Left at Material's defaults (for the caller to
		// fill in manually later): alpha_mode/alpha_cutoff, double_sided, normal_texture, and
		// occlusion_texture -- base_color_texture and metallic_roughness_texture are handled just
		// below, they're the two textures Material currently supports.
		mat.m_albedoColor = float3((float)pbr.base_color_factor[0], (float)pbr.base_color_factor[1], (float)pbr.base_color_factor[2]);
		mat.m_emissiveColor = float3((float)gltfMat.emissive_factor[0], (float)gltfMat.emissive_factor[1], (float)gltfMat.emissive_factor[2]);
		mat.m_roughness = (float)pbr.roughness_factor;
		mat.m_metallic = (float)pbr.metallic_factor;
		// No texture until proven otherwise below -- matches the convention already used for
		// hand-authored materials with no texture (see App.cpp's m0).
		mat.m_albedoTextureFade = 0.0f;

		mat.m_hndTexture = tg3_load_embedded_texture(model, vulkanContext, pbr.base_color_texture.index, debugName + " (base color)", /*isSRGB=*/true);
		if (mat.m_hndTexture.IsValid())
			mat.m_albedoTextureFade = 1.0f;

		mat.m_hndMetallicRoughnessTexture = tg3_load_embedded_texture(model, vulkanContext, pbr.metallic_roughness_texture.index, debugName + " (metallic-roughness)", /*isSRGB=*/false);

		outMaterialHandles.push_back(g_MaterialManager.push_back(mat));
	}

	return true;
}