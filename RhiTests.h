#pragma once

// Documents (via executable tests) the RHI/Vulkan layer's object lifecycle contract:
// utils::Context / utils::ResourceAllocator (extern/vk_framework.h) and the engine's
// Buffer class (Buffer.h) -- what state a fresh object has, what a successful init()
// produces, and what deinit() must leave behind. See TestMode.h for how this suite is
// invoked.
//
// Unlike GlmMathTests, this suite creates a real (windowless) Vulkan instance/device, so
// it needs a working Vulkan-capable GPU -- the same requirement the app itself has. It
// deliberately never exercises invalid Vulkan usage: extern/vk_framework.h's VK_CHECK/
// ASSERT macros abort the process on any real driver error (by design, matching the
// shipping app), and the validation layer's error callback calls __debugbreak() -- so an
// intentionally-invalid call here would crash the test run instead of reporting a clean
// FAIL. The suite stays within valid usage and reports on state, not on provoking errors.

namespace tests { class TestReport; }

namespace tests
{
	void RunRhiTests(TestReport& report);
}
