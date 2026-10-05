#include "global.h"	// include first in any cpp

#include "TestMode.h"
#include "TestFramework.h"
#include "GlmMathTests.h"
#include "RhiTests.h"
#include "AppendArenaTests.h"
#include "SceneTests.h"
#include "MaterialTextureTests.h"
#include "ObjManagerTests.h"

#include <windows.h>	// OutputDebugStringA
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace tests
{
	bool RunTestModeIfRequested(int argc, char** argv, int& outExitCode)
	{
		bool wantTest = false;
		for (int i = 1; i < argc; ++i)
			if (strcmp(argv[i], "--test") == 0)
				wantTest = true;

		if (!wantTest)
			return false;

		// same OBS-capture-layer workaround main() applies before creating a real window --
		// RhiTests creates a real (headless) Vulkan instance/device too, and must set this
		// before any Vulkan call for it to take effect.
		SetEnvironmentVariableA("DISABLE_VULKAN_OBS_CAPTURE", "1");

		const char* reportFileName = "VVerseTestReport.txt";

		TestReport report;
		report.SetOutputFile(reportFileName);

		printf("VVerse test mode -- running all suites.\n\n");

		RunGlmMathTests(report);
		RunRhiTests(report);
		RunAppendArenaTests(report);
		RunSceneTests(report);
		RunMaterialTextureTests(report);
		RunObjManagerTests(report);

		report.PrintSummary();

		outExitCode = report.AllPassed() ? 0 : 1;

		// x64 builds run with SubSystem=Windows (no console attached), so stdout is only
		// visible if the caller redirected it; OutputDebugString always reaches a debugger
		// (Visual Studio Output window, DebugView, etc.), so this is how you find the report
		// when the exe was just double-clicked or launched under the debugger with F5.
		std::string absPath = std::filesystem::absolute(reportFileName).string();
		char msg[512];
		snprintf(msg, sizeof(msg), "VVerse test mode finished: %s (%d/%d passed). Report written to: %s\n",
			report.AllPassed() ? "ALL TESTS PASSED" : "TESTS FAILED", report.GetPassCount(),
			report.GetPassCount() + report.GetFailCount(), absPath.c_str());
		OutputDebugStringA(msg);
		printf("%s", msg);

		return true;
	}

	bool WantsSelfTestMode(int argc, char** argv)
	{
		for (int i = 1; i < argc; ++i)
			if (strcmp(argv[i], "--selftest") == 0)
				return true;
		return false;
	}
}
