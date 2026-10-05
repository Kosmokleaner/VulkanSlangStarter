#include "global.h"	// include first in any cpp

#include "MaterialTextureTests.h"
#include "TestFramework.h"
#include "Material.h"
#include "Texture.h"
#include "AppendArena.h"

#include <utility>

namespace tests
{
	static const char* kSuiteMaterialTexture = "MaterialTexture";

	// Same idea as SceneTests.cpp's ScopedFreshSceneObjects: keeps each test's objects isolated
	// from whatever is already in the global managers, independent of any manager/index reuse.
	struct ScopedFreshTextureManager
	{
		CObjManager<CTexHandle, Texture> previous;
		ScopedFreshTextureManager()
		{
			previous = std::move(g_TextureManager);
			g_TextureManager = CObjManager<CTexHandle, Texture>();
			g_TextureManager.InitOnce(16);
		}
		~ScopedFreshTextureManager() { g_TextureManager = std::move(previous); }
	};
	struct ScopedFreshMaterialManager
	{
		CObjManager<CMatHandle, Material> previous;
		ScopedFreshMaterialManager()
		{
			previous = std::move(g_MaterialManager);
			g_MaterialManager = CObjManager<CMatHandle, Material>();
			g_MaterialManager.InitOnce(16);
		}
		~ScopedFreshMaterialManager() { g_MaterialManager = std::move(previous); }
	};

	static void RunProcess(TokenAppendArena& arena)
	{
		HandleRemapTable remapping;
		ArenaTokenRange range;
		range.arena = &arena;
		range.size = arena.size();
		process(range, remapping);
	}

	// saveTextures() writes m_name via appendString() (blob-referenced, like Scene's Transform)
	// and m_loadFromFile via a direct value -- both dispatched at load through ATokenCmdSetMember's
	// type==2 branch (AppendArena.cpp). Doesn't call Texture::loadFromFile()/touch the GPU; that's
	// a separate concern needing a real Vulkan device (see RhiTests.h).
	static void Test_SaveTexturesRoundTripsNameAndLoadFromFile(TestReport& report)
	{
		TokenAppendArena arena;
		{
			ScopedFreshTextureManager saveScope;
			Texture tex;
			tex.m_name = "resources/test_image.jpg";
			tex.m_loadFromFile = true;
			g_TextureManager.push_back(tex);

			saveTextures(arena);
		}

		ScopedFreshTextureManager loadScope;
		RunProcess(arena);

		uint32 count = g_TextureManager.GetElementCount();
		report.Check(kSuiteMaterialTexture, "texture_load_recreates_one_object", count == 1,
			"after saveTextures() of 1 Texture then load, g_TextureManager must contain 1 object, got " + std::to_string(count));

		if (count == 1)
		{
			Texture& loaded = g_TextureManager.GetElementByIterator(0);
			report.Check(kSuiteMaterialTexture, "texture_load_restores_name", loaded.m_name == "resources/test_image.jpg",
				"reloaded Texture::m_name must match what was saved, got '" + loaded.m_name + "'");
			report.Check(kSuiteMaterialTexture, "texture_load_restores_loadFromFile", loaded.m_loadFromFile == true,
				"reloaded Texture::m_loadFromFile must match what was saved (true), got " + std::to_string(loaded.m_loadFromFile));
		}
	}

	// saveMaterials() writes m_name via appendString(), m_albedoColor/m_emissiveColor via
	// appendObject() (both blob-referenced), and the scalar/handle members as direct values --
	// covers every ATokenCmdSetMember memberIndex (0..8) Material.cpp's saveMaterials() emits.
	// Also verifies a Material's texture handle resolves to the co-reloaded Texture, matching
	// saveScene()'s real save order (saveTextures() then saveMaterials()).
	static void Test_SaveMaterialsRoundTripsFieldsAndTextureHandle(TestReport& report)
	{
		TokenAppendArena arena;
		{
			ScopedFreshTextureManager textureSaveScope;
			ScopedFreshMaterialManager materialSaveScope;

			Texture tex;
			tex.m_name = "resources/albedo.png";
			tex.m_loadFromFile = true;
			CTexHandle hndTex = g_TextureManager.push_back(tex);

			Material mat;
			mat.m_name = "shiny";
			mat.m_albedoColor = float3(0.25f, 0.5f, 0.75f);
			mat.m_emissiveColor = float3(1.0f, 0.0f, 0.0f);
			mat.m_roughness = 0.2f;
			mat.m_metallic = 0.8f;
			mat.m_albedoTextureFade = 0.6f;
			mat.m_hndTexture = hndTex;
			g_MaterialManager.push_back(mat);

			saveTextures(arena);
			saveMaterials(arena);
		}

		ScopedFreshTextureManager textureLoadScope;
		ScopedFreshMaterialManager materialLoadScope;
		RunProcess(arena);

		uint32 texCount = g_TextureManager.GetElementCount();
		uint32 matCount = g_MaterialManager.GetElementCount();
		report.Check(kSuiteMaterialTexture, "material_load_recreates_texture_and_material", texCount == 1 && matCount == 1,
			"after saveTextures()+saveMaterials() of 1 each then load, expected 1 texture and 1 material, got "
			+ std::to_string(texCount) + " textures, " + std::to_string(matCount) + " materials");

		if (texCount == 1 && matCount == 1)
		{
			Material& loaded = g_MaterialManager.GetElementByIterator(0);
			CTexHandle reloadedTexHandle = g_TextureManager.GetHandleByIterator(0);

			report.Check(kSuiteMaterialTexture, "material_load_restores_name", loaded.m_name == "shiny",
				"reloaded Material::m_name must match what was saved, got '" + loaded.m_name + "'");
			report.Check(kSuiteMaterialTexture, "material_load_restores_albedo",
				NearlyEqual(loaded.m_albedoColor, float3(0.25f, 0.5f, 0.75f)),
				"reloaded Material::m_albedoColor must match what was saved, got " + ToString(loaded.m_albedoColor));
			report.Check(kSuiteMaterialTexture, "material_load_restores_emissive",
				NearlyEqual(loaded.m_emissiveColor, float3(1.0f, 0.0f, 0.0f)),
				"reloaded Material::m_emissiveColor must match what was saved, got " + ToString(loaded.m_emissiveColor));
			report.Check(kSuiteMaterialTexture, "material_load_restores_roughness", NearlyEqual(loaded.m_roughness, 0.2f),
				"reloaded Material::m_roughness must match what was saved, got " + std::to_string(loaded.m_roughness));
			report.Check(kSuiteMaterialTexture, "material_load_restores_metallic", NearlyEqual(loaded.m_metallic, 0.8f),
				"reloaded Material::m_metallic must match what was saved, got " + std::to_string(loaded.m_metallic));
			report.Check(kSuiteMaterialTexture, "material_load_restores_albedoTextureFade", NearlyEqual(loaded.m_albedoTextureFade, 0.6f),
				"reloaded Material::m_albedoTextureFade must match what was saved, got " + std::to_string(loaded.m_albedoTextureFade));
			report.Check(kSuiteMaterialTexture, "material_load_restores_texture_handle",
				loaded.m_hndTexture.getRaw() == reloadedTexHandle.getRaw(),
				"reloaded Material::m_hndTexture must point at the reloaded Texture object");
		}
	}

	void RunMaterialTextureTests(TestReport& report)
	{
		Test_SaveTexturesRoundTripsNameAndLoadFromFile(report);
		Test_SaveMaterialsRoundTripsFieldsAndTextureHandle(report);
	}
}
