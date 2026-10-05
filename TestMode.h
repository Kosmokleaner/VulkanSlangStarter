#pragma once

// Unsupervised test mode: runs all registered test suites (see GlmMathTests.h and future
// suites) with no window, no Vulkan device, and no user interaction, then exits. Meant to
// be checked from the command line or by an AI assistant reading the output/report file --
// results are plain PASS/FAIL lines plus a final summary, so no test framework needs to be
// installed to read them. Only covers CPU-side logic that doesn't need a GPU (save/load byte
// round-trips, math, ObjManager, ...) -- for anything that needs to actually render and look
// right, see --selftest below instead.
//
// Usage: VVerse.exe --test
//
// A second, different kind of automated run: VVerse.exe --selftest -- a deterministic scripted
// pass through the real app (real window, real Vulkan device: setup1/setup2/save/load, each
// rendered and screenshotted, compared against checked-in gold images) instead of headless
// logic-only checks. See App::runSelfTest() (App.cpp) for the actual script
// and TestData/ for its gold images/test scene files.

namespace tests
{
	// @return true if argv requested test mode (in which case it was already run to
	//   completion and the process should exit with the returned code instead of starting
	//   the normal application)
	bool RunTestModeIfRequested(int argc, char** argv, int& outExitCode);

	// @return true if argv requested the OTHER kind of automated test -- VVerse.exe --selftest --
	// a deterministic scripted run through the REAL app (real window, real Vulkan device; see
	// App::runSelfTest(), App.cpp) instead of RunTestModeIfRequested()'s
	// headless one. Only parses argv; the caller (main()) still constructs App and calls
	// runSelfTest() on it itself, since this function has no app instance to drive.
	bool WantsSelfTestMode(int argc, char** argv);
}
