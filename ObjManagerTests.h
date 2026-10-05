#pragma once

// Verifies CObjManager's element lifetime handling: RemoveByIndex() must run the removed
// element's destructor, and push_back() into a freed slot must run the constructor again
// (not just assign over the stale object). See TestMode.h for how this suite is invoked.

namespace tests { class TestReport; }

namespace tests
{
	void RunObjManagerTests(TestReport& report);
}
