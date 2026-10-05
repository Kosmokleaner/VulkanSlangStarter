#include "global.h"	// include first in any cpp

#include "SceneTests.h"
#include "TestFramework.h"
#include "Scene.h"
#include "AppendArena.h"
#include "RenderComponent.h"

#include <utility>

namespace tests
{
	static const char* kSuiteScene = "Scene";

	struct ScopedFreshSceneObjects
	{
		CObjManager<CEntityHandle, Entity> previous;

		ScopedFreshSceneObjects()
		{
			previous = std::move(g_scene.m_sceneObjects);
			g_scene.m_sceneObjects = CObjManager<CEntityHandle, Entity>();
			g_scene.m_sceneObjects.InitOnce(16);
		}
		~ScopedFreshSceneObjects()
		{
			g_scene.m_sceneObjects = std::move(previous);
		}
	};

	// g_MeshManager only ever holds raw Mesh* (never deleted anywhere, see Mesh.h) -- matches
	// that same never-free approach here instead of introducing a different lifetime rule just
	// for tests.
	struct ScopedFreshMeshManager
	{
		CObjManager<CRenderCompHandle, RenderComponent> previous;

		ScopedFreshMeshManager()
		{
			previous = std::move(g_MeshManager);
			g_MeshManager = CObjManager<CRenderCompHandle, RenderComponent>();
			g_MeshManager.InitOnce(16);
		}
		~ScopedFreshMeshManager()
		{
			g_MeshManager = std::move(previous);
		}
	};

	// A minimal procedural (m_loadFromFile == false) mesh: saveMeshes() serializes its raw
	// m_sourceIndices/m_sourceVertices for this kind, and getVector()/appendBytes() both assert
	// on an empty vector, so this needs at least one triangle's worth of data.
	static RenderComponent MakeTestMesh()
	{
		RenderComponent mesh;
		mesh.m_filePath = "test-procedural-mesh";
		mesh.m_loadFromFile = false;
		mesh.m_sourceIndices = { 0, 1, 2 };
		mesh.m_sourceVertices.resize(3);
		return mesh;
	}

	static void Test_SaveWritesNonEmptyArena(TestReport& report)
	{
		ScopedFreshSceneObjects scope;

		Entity obj;
		obj.transform.SetPos(float3(1.0f, 2.0f, 3.0f));
		obj.hndRenderComp.setRaw(0x0002000000010000ull);	// arbitrary but distinctive, no real Mesh needed
		g_scene.m_sceneObjects.push_back(obj);

		TokenAppendArena arena;
		g_scene.save(arena);

		report.Check(kSuiteScene, "save_writes_tokens_for_one_object", arena.size() > 0,
			"Scene::save() with one Entity in m_sceneObjects must append at least one token to the arena, got size "
			+ std::to_string(arena.size()));
	}

	// The full save -> load round trip: build a simple scene (plus the real meshes its objects
	// reference, since ATokenCmdSetMember now resolves a SceneObject's hndMesh through
	// HandleRemapTable rather than passing the saved raw handle through unchanged -- see
	// AppendArena.h's HandleRemapTable comment), save it to an in-memory arena (never written to
	// disk -- the arena's token stream is the API surface under test), then load it into fresh
	// managers and check the reloaded objects match what was saved.
	static void Test_SaveThenLoadRoundTripsSceneObjects(TestReport& report)
	{
		Entity a;
		a.transform.SetPos(float3(10.0f, 20.0f, 30.0f));

		Entity b;
		b.transform.SetPos(float3(-1.0f, 0.5f, 4.25f));

		TokenAppendArena arena;
		{
			ScopedFreshMeshManager meshSaveScope;
			ScopedFreshSceneObjects sceneSaveScope;

			a.hndRenderComp = g_MeshManager.push_back(MakeTestMesh());
			b.hndRenderComp = g_MeshManager.push_back(MakeTestMesh());

			g_scene.m_sceneObjects.push_back(a);
			g_scene.m_sceneObjects.push_back(b);

			// real save order (see App.cpp's saveScene()): meshes before the scene
			// objects that reference them, so their handles are already remapped on load by the
			// time the scene objects' ATokenCmdSetMember commands run
			saveMeshes(arena);
			g_scene.save(arena);
		}

		ScopedFreshMeshManager meshLoadScope;
		ScopedFreshSceneObjects sceneLoadScope;
		{
			HandleRemapTable remapping;
			ArenaTokenRange range;

			range.arena = &arena;
			range.size = arena.size();

			process(range, remapping);
		}

		uint32 loadedCount = g_scene.m_sceneObjects.GetElementCount();
		report.Check(kSuiteScene, "load_recreates_same_object_count", loadedCount == 2,
			"after save() of 2 SceneObjects then load() into a fresh scene, m_sceneObjects must contain 2 objects, got "
			+ std::to_string(loadedCount));

		uint32 loadedMeshCount = g_MeshManager.GetElementCount();
		report.Check(kSuiteScene, "load_recreates_same_mesh_count", loadedMeshCount == 2,
			"after saveMeshes() of 2 meshes then load(), g_MeshManager must contain 2 objects, got "
			+ std::to_string(loadedMeshCount));

		if (loadedCount == 2 && loadedMeshCount == 2)
		{
			Entity& loadedA = g_scene.m_sceneObjects.GetElementByIterator(0);
			Entity& loadedB = g_scene.m_sceneObjects.GetElementByIterator(1);

			// the reloaded mesh for a given Entity can land at a different manager slot than
			// it was saved at (see HandleRemapTable) -- distinguish the two meshes by their
			// reloaded m_filePath, then check each Entity's hndMesh points at its own mesh,
			// not the other one or an invalid handle
			CRenderCompHandle reloadedMeshA = g_MeshManager.GetHandleByIterator(0);
			CRenderCompHandle reloadedMeshB = g_MeshManager.GetHandleByIterator(1);

			report.Check(kSuiteScene, "load_restores_first_object_position",
				NearlyEqual(loadedA.transform.GetPos(), a.transform.GetPos()),
				"first reloaded Entity's transform.position must match what was saved ("
				+ ToString(a.transform.GetPos()) + "), got " + ToString(loadedA.transform.GetPos()));
			report.Check(kSuiteScene, "load_restores_first_object_mesh_handle",
				loadedA.hndRenderComp.getRaw() == reloadedMeshA.getRaw(),
				"first reloaded Entity's hndMesh must point at the reloaded mesh it was saved with, "
				"translated through HandleRemapTable rather than left as the raw saved handle");

			report.Check(kSuiteScene, "load_restores_second_object_position",
				NearlyEqual(loadedB.transform.GetPos(), b.transform.GetPos()),
				"second reloaded Entity's transform.position must match what was saved ("
				+ ToString(b.transform.GetPos()) + "), got " + ToString(loadedB.transform.GetPos()));
			report.Check(kSuiteScene, "load_restores_second_object_mesh_handle",
				loadedB.hndRenderComp.getRaw() == reloadedMeshB.getRaw(),
				"second reloaded Entity's hndMesh must point at the reloaded mesh it was saved with, "
				"translated through HandleRemapTable rather than left as the raw saved handle");
		}
	}

	// The scenario that used to break silently: the manager being loaded into is *not* virgin
	// (some objects already live in it, as when g_MeshManager keeps a few meshes across a scene
	// reload) -- push_back() then hands new meshes indices that don't line up with what got saved,
	// so a SceneObject's hndMesh must be resolved through HandleRemapTable, not through the saved
	// raw handle happening to still point at the right slot.
	static void Test_LoadRemapsMeshHandleWhenManagerNotVirgin(TestReport& report)
	{
		Entity obj;
		obj.transform.SetPos(float3(5.0f, 6.0f, 7.0f));

		TokenAppendArena arena;
		{
			ScopedFreshMeshManager meshSaveScope;
			ScopedFreshSceneObjects sceneSaveScope;

			obj.hndRenderComp = g_MeshManager.push_back(MakeTestMesh());
			g_scene.m_sceneObjects.push_back(obj);

			saveMeshes(arena);
			g_scene.save(arena);
		}

		ScopedFreshMeshManager meshLoadScope;
		ScopedFreshSceneObjects sceneLoadScope;

		// unlike the round-trip test above, put an unrelated mesh in the manager *before*
		// loading -- push_back() for the loaded mesh must land at index 1, not index 0 like it
		// did when it was saved
		CRenderCompHandle preExisting = g_MeshManager.push_back(MakeTestMesh());

		{
			HandleRemapTable remapping;
			ArenaTokenRange range;

			range.arena = &arena;
			range.size = arena.size();

			process(range, remapping);
		}

		uint32 loadedMeshCount = g_MeshManager.GetElementCount();
		report.Check(kSuiteScene, "nonvirgin_load_recreates_preexisting_plus_loaded_mesh", loadedMeshCount == 2,
			"loading one saved mesh into a manager that already held one other mesh must leave 2 meshes total, got "
			+ std::to_string(loadedMeshCount));

		uint32 loadedObjCount = g_scene.m_sceneObjects.GetElementCount();
		if (loadedMeshCount == 2 && loadedObjCount == 1)
		{
			Entity& loadedObj = g_scene.m_sceneObjects.GetElementByIterator(0);

			// the loaded mesh must have landed at whatever slot push_back() actually gave it --
			// not necessarily index 0 -- so find it by process of elimination against the
			// pre-existing one, then check hndMesh points there and nowhere else
			CRenderCompHandle handleA = g_MeshManager.GetHandleByIterator(0);
			CRenderCompHandle handleB = g_MeshManager.GetHandleByIterator(1);
			CRenderCompHandle loadedMeshHandle = (handleA.getRaw() == preExisting.getRaw()) ? handleB : handleA;

			report.Check(kSuiteScene, "nonvirgin_load_remaps_mesh_handle_correctly",
				loadedObj.hndRenderComp.getRaw() == loadedMeshHandle.getRaw() && loadedObj.hndRenderComp.getRaw() != preExisting.getRaw(),
				"Entity::hndMesh must be translated to wherever the loaded mesh actually landed, not left as "
				"its saved (now-stale) handle and not pointing at the pre-existing mesh either");
		}
	}

	// Save/load must not reorder scene objects -- confirmed live as a --selftest gold-image mismatch
	// (0.77% of pixels, moderate magnitude, localized) between a freshly-built BSP scene and the same
	// scene reloaded from disk: updateLights() (App.cpp) builds shaderio::Light[] by
	// iterating g_scene.m_sceneObjects via GetElementByIterator(), and ray.slang's PickUniformRandomLight
	// picks by INDEX into that array -- so if load() ever hands two objects back in a different
	// relative order than they were saved in, the same per-pixel random draw lands on a different
	// light, changing the render even though every individual object's own data round-tripped
	// correctly (which Test_SaveThenLoadRoundTripsSceneObjects above already covers, but only ever
	// with 2 objects addressed directly by iterator index -- not a real test of ORDER, just of value,
	// and easy to pass by 50/50 luck). More objects here specifically to make a reordering bug show
	// up reliably instead of by chance.
	static void Test_SaveThenLoadPreservesIterationOrder(TestReport& report)
	{
		const int kCount = 8;

		TokenAppendArena arena;
		{
			ScopedFreshSceneObjects sceneSaveScope;

			for (int i = 0; i < kCount; ++i)
			{
				Entity obj;
				// Distinct, order-revealing position per object (no two objects share a coordinate)
				// -- identifies which original object a given reloaded slot came from without
				// relying on iteration order to already be correct.
				obj.transform.SetPos(float3(float(i), 0.0f, 0.0f));
				g_scene.m_sceneObjects.push_back(obj);
			}

			g_scene.save(arena);
		}

		ScopedFreshSceneObjects sceneLoadScope;
		{
			HandleRemapTable remapping;
			ArenaTokenRange range;
			range.arena = &arena;
			range.size = arena.size();
			process(range, remapping);
		}

		uint32 loadedCount = g_scene.m_sceneObjects.GetElementCount();
		report.Check(kSuiteScene, "order_load_recreates_same_object_count", loadedCount == (uint32)kCount,
			"after save() of " + std::to_string(kCount) + " SceneObjects then load(), m_sceneObjects must contain "
			+ std::to_string(kCount) + " objects, got " + std::to_string(loadedCount));

		if (loadedCount != (uint32)kCount)
			return;

		bool orderPreserved = true;
		std::string detail;
		for (uint32 i = 0; i < loadedCount && orderPreserved; ++i)
		{
			float x = g_scene.m_sceneObjects.GetElementByIterator(i).transform.GetPos().x;
			if (!NearlyEqual(x, (float)i))
			{
				orderPreserved = false;
				detail = "iterator " + std::to_string(i) + " holds the object saved at position x=" + std::to_string(x)
					+ ", expected the one saved at x=" + std::to_string(i) + " (i.e. objects came back in a different order than they were saved in)";
			}
		}

		report.Check(kSuiteScene, "save_then_load_preserves_iteration_order", orderPreserved,
			orderPreserved
				? "reloaded " + std::to_string(kCount) + " objects came back in the exact order they were saved in"
				: detail);
	}

	void RunSceneTests(TestReport& report)
	{
		Test_SaveWritesNonEmptyArena(report);
		Test_SaveThenLoadRoundTripsSceneObjects(report);
		Test_LoadRemapsMeshHandleWhenManagerNotVirgin(report);
		Test_SaveThenLoadPreservesIterationOrder(report);
	}
}
