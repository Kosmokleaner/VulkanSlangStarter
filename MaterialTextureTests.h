#pragma once

// Verifies saveTextures()/saveMaterials() (Texture.cpp/Material.cpp) round-trip through an
// in-memory TokenAppendArena via process() -- the same token-level API Scene::save()/load() use
// (see SceneTests.h). Does not exercise Texture::loadFromFile()/GPU upload -- that needs a real
// Vulkan device (see RhiTests.h) and is a separate concern from whether the token stream itself
// carries the right values.

namespace tests { class TestReport; }

namespace tests
{
	void RunMaterialTextureTests(TestReport& report);
}
