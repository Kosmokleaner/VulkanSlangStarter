#pragma once
#include "ObjManager.h"
#include <string>

class TokenAppendArena;
struct ATokenCmdSetMember;
struct ArenaTokenRange;
class HandleRemapTable;

class Material
{
	// for access in global array
	CMatHandle m_hndMaterial;

public:
	// How this material expects to be lit -- currently just a data/UI tag (see ImGuiMaterial(),
	// App.cpp, and its own typeUI combo string below); nothing reads it to change
	// lighting behavior yet, but it's the natural place to branch from once something does (e.g.
	// skipping real-time BRDF evaluation for Lightmapped surfaces in favor of their baked lightmap).
	enum class EType : uint32
	{
		PBR,          // real-time BRDF lighting (point lights, IBL, ...) -- the common case
		Lightmapped,  // pre-baked static lighting (m_hndLightmap) -- BSP content, see 3DBSPLevel.cpp
	};
	static constexpr char* typeUI = (char*)"PBR\0Lightmapped\0";

	// [0]
	std::string m_name;
	// [11]
	EType m_type = EType::PBR;

	// [1] 0..1
	float3 m_albedoColor = float3(1.0f, 1.0f, 1.0f);
	// [2] 0..1
	float3 m_emissiveColor = {};
	// [3] 0..1
	float m_roughness = 0.1f;
	// [4] 0..1
	float m_metallic = 0.0f;
	// [5] 0:no texture .. 1:fully visible
	float m_albedoTextureFade = 1.0f;
	// [6] the glTF base color texture (or a hand-assigned one), managed by g_TextureManager; invalid
	// (IsValid()==false) if this material has no per-material texture -- see Texture.h
	CTexHandle m_hndTexture;
	// [7] the glTF metallic-roughness texture (G=roughness, B=metallic per the glTF spec), managed by
	// g_TextureManager; invalid (IsValid()==false) if this material has none -- when valid, it's
	// multiplied with m_roughness/m_metallic rather than blended by a fade (glTF has no such factor
	// for this texture).
	CTexHandle m_hndMetallicRoughnessTexture;
	// [8]
	CTexHandle m_hndLightmap;
	// [9] 0 = disabled, >0 = darker albedo suppresses specular too -- see MaterialData::
	// specularOcclusionFromAlbedo's comment (shader_io.h) for the full rationale/formula. 0 by
	// default (most materials are real PBR content where this wouldn't be physically correct);
	// BSP-generated materials set this to 0.25 (3DBSPLevel.cpp).
	float m_specOcclusionFromAlbedo = 0.0f;
	// [10] Tangent-space normal map (glTF's, a hand-assigned one, or a BSP-generated one -- see
	// GenerateNormalMapFromLuminance, Texture.h). Invalid (IsValid()==false) is the common case
	// (most materials have none), NOT resolved to a flat-normal fallback here -- that happens once,
	// per frame, on the GPU-side MaterialData fill (see g_hndDefaultNormalMap, App.cpp),
	// same reasoning as textureId's kInvalidTextureId sentinel elsewhere: keeps this a plain
	// "has one or doesn't" handle instead of every construction site needing to know about the
	// engine default.
	CTexHandle m_hndNormalMap;
	// [12] 0..16, multiplies m_emissiveColor before it reaches the shader (see MaterialData::
	// emissiveIntensity, shader_io.h) -- lets an emissive material read as a genuinely bright light
	// source instead of being capped to the same 0..1 range as a plain albedo color. 1.0 = off.
	float m_emissiveIntensity = 1.0f;

	// for access in global array
	CMatHandle getHandle() const { ASSERT(m_hndMaterial.IsValid()); return m_hndMaterial; }
};

extern CObjManager<CMatHandle, Material> g_MaterialManager;

void saveMaterials(TokenAppendArena& arena);
// Sets a single member of ref (cmd->memberIndex says which) while loading a saved scene -- called
// from AppendArena.cpp's ATokenCmdSetMember::execute(), once per member, once per material. See
// AppendArena.h's top-of-file backward-compatibility comment before changing memberIndex mappings.
void load(Material& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping);
