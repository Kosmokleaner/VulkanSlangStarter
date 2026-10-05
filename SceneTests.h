#pragma once

// Verifies Scene::save()/Scene::load() round-trip scene objects through an in-memory
// TokenAppendArena (no disk I/O -- the arena's token stream is the API surface under test).
// See TestMode.h for how this suite is invoked.

namespace tests { class TestReport; }

namespace tests
{
	void RunSceneTests(TestReport& report);
}
