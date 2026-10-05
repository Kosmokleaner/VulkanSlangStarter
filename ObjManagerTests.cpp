#include "global.h"	// include first in any cpp

#include "ObjManagerTests.h"
#include "TestFramework.h"
#include "ObjManager.h"

#include <string>

namespace tests
{
	static const char* kSuiteObjManager = "ObjManager";

	// A type whose constructors/destructor record every call, so CObjManager's element
	// lifetime handling can be observed directly instead of just inferred from reading its
	// implementation. push_back() copies its argument in, so the copy constructor is the
	// one that matters for "reused slot" checks below.
	struct LifecycleProbe
	{
		static int s_constructCount;
		static int s_destructCount;

		int value = -1;

		LifecycleProbe() { ++s_constructCount; }
		explicit LifecycleProbe(int inValue) : value(inValue) { ++s_constructCount; }
		LifecycleProbe(const LifecycleProbe& other) : value(other.value) { ++s_constructCount; }
		LifecycleProbe& operator=(const LifecycleProbe& other) { value = other.value; return *this; }
		~LifecycleProbe() { ++s_destructCount; }

		static void Reset() { s_constructCount = 0; s_destructCount = 0; }
	};

	int LifecycleProbe::s_constructCount = 0;
	int LifecycleProbe::s_destructCount = 0;

	static void Test_RemoveByIndexCallsDestructor(TestReport& report)
	{
		LifecycleProbe::Reset();

		CObjManager<CObjHandle, LifecycleProbe> man;
		man.InitOnce(4);	// vector::resize() default-constructs one placeholder per slot

		report.Check(kSuiteObjManager, "initonce_constructs_placeholders", LifecycleProbe::s_constructCount == 4,
			"InitOnce(4) must default-construct exactly 4 placeholder elements (one per slot), got "
			+ std::to_string(LifecycleProbe::s_constructCount));

		LifecycleProbe source(42);
		CObjHandle hnd = man.push_back(source);

		LifecycleProbe* p = man.GetElementPtr(hnd);
		report.Check(kSuiteObjManager, "pushback_reads_back_value", p && p->value == 42,
			"push_back()'d value must be readable back through GetElementPtr()");

		int destructsBeforeRemove = LifecycleProbe::s_destructCount;

		man.RemoveByIndex(hnd.Index);

		int destructsAfterRemove = LifecycleProbe::s_destructCount;

		report.Check(kSuiteObjManager, "removebyindex_calls_destructor",
			destructsAfterRemove == destructsBeforeRemove + 1,
			"RemoveByIndex() must call the removed element's destructor exactly once, destruct count went from "
			+ std::to_string(destructsBeforeRemove) + " to " + std::to_string(destructsAfterRemove));
	}

	static void Test_ReusedSlotCallsConstructor(TestReport& report)
	{
		LifecycleProbe::Reset();

		CObjManager<CObjHandle, LifecycleProbe> man;
		man.InitOnce(2);

		LifecycleProbe first(1);
		CObjHandle a = man.push_back(first);

		man.RemoveByIndex(a.Index);

		int constructsBeforeReuse = LifecycleProbe::s_constructCount;

		LifecycleProbe second(2);
		CObjHandle b = man.push_back(second);

		int constructsAfterReuse = LifecycleProbe::s_constructCount;

		// one construction for `second` itself (the local variable above) plus one more for
		// push_back()'s own copy-construction into the reused slot -- if push_back() instead
		// assigned over the stale object (the old behavior), the slot's constructor would not
		// run again here
		report.Check(kSuiteObjManager, "reused_slot_constructs_new_object",
			constructsAfterReuse == constructsBeforeReuse + 2,
			"constructing `second` and then push_back()'ing it into a freed slot must run LifecycleProbe's "
			"constructor twice more (once for the local, once for push_back()'s copy into the slot), construct "
			"count went from " + std::to_string(constructsBeforeReuse) + " to " + std::to_string(constructsAfterReuse));

		LifecycleProbe* p = man.GetElementPtr(b);
		report.Check(kSuiteObjManager, "reused_slot_holds_new_value",
			b.Index == a.Index && p && p->value == 2,
			"the reused slot must hold the newly pushed value (2), not the removed one (1)");
	}

	// GetHandleByIterator(i) and GetElementByIterator(i) must always refer to the same element --
	// callers (e.g. the save*() functions) pair them up to write a (handle, value) token. Removing
	// a non-last element forces CObjManager to swap the last used element into the removed slot's
	// iterator position (SwapInUsedAndUsed), which is what exposes a handle whose Index isn't
	// actually the element's position in Elements[] (see GetHandleByIterator's comment).
	static void Test_GetHandleByIteratorMatchesElementAfterRemoval(TestReport& report)
	{
		CObjManager<CObjHandle, int> man;
		man.InitOnce(4);

		CObjHandle a = man.push_back(1);
		man.push_back(2);
		man.push_back(3);

		// remove the FIRST element, not the last -- this is what forces the swap
		man.Remove(a);

		bool allMatch = true;
		for (uint32 i = 0, count = man.GetElementCount(); i < count; ++i)
		{
			CObjHandle hnd = man.GetHandleByIterator(i);
			int* viaHandle = man.GetElementPtr(hnd);
			int* viaIterator = &man.GetElementByIterator(i);
			if (viaHandle != viaIterator)
				allMatch = false;
		}

		report.Check(kSuiteObjManager, "gethandlebyiterator_matches_element_after_removal", allMatch,
			"GetHandleByIterator(i) must resolve, via GetElementPtr(), to the exact same element "
			"GetElementByIterator(i) returns, even after a removal has shuffled the free list");
	}

	void RunObjManagerTests(TestReport& report)
	{
		Test_RemoveByIndexCallsDestructor(report);
		Test_ReusedSlotCallsConstructor(report);
		Test_GetHandleByIteratorMatchesElementAfterRemoval(report);
	}
}
