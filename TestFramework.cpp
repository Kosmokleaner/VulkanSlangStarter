#include "global.h"	// include first in any cpp

#include "TestFramework.h"

#include <cstdio>

namespace tests
{
	void TestReport::Emit(const std::string& line)
	{
		printf("%s\n", line.c_str());

		if (m_outFile.is_open())
			m_outFile << line << std::endl;
	}

	void TestReport::Check(const std::string& suite, const std::string& name, bool condition, const std::string& detail)
	{
		TestResult r;
		r.suite = suite;
		r.name = name;
		r.passed = condition;
		r.detail = detail;
		m_results.push_back(r);

		char line[1024];
		snprintf(line, sizeof(line), "[%s] %s / %s -- %s", condition ? "PASS" : "FAIL", suite.c_str(), name.c_str(), detail.c_str());
		Emit(line);
	}

	void TestReport::SetOutputFile(const std::string& path)
	{
		m_outFile.open(path, std::ios::out);
	}

	int TestReport::GetFailCount() const
	{
		int n = 0;
		for (const TestResult& r : m_results)
			if (!r.passed)
				n++;
		return n;
	}

	int TestReport::GetPassCount() const
	{
		return (int)m_results.size() - GetFailCount();
	}

	void TestReport::PrintSummary()
	{
		Emit("");
		Emit("==================== TEST SUMMARY ====================");

		char line[256];
		snprintf(line, sizeof(line), "%d passed, %d failed, %d total", GetPassCount(), GetFailCount(), (int)m_results.size());
		Emit(line);

		if (GetFailCount() > 0)
		{
			Emit("Failures:");
			for (const TestResult& r : m_results)
			{
				if (!r.passed)
				{
					char failLine[1024];
					snprintf(failLine, sizeof(failLine), "  - %s / %s -- %s", r.suite.c_str(), r.name.c_str(), r.detail.c_str());
					Emit(failLine);
				}
			}
		}

		Emit(AllPassed() ? "RESULT: ALL TESTS PASSED" : "RESULT: TESTS FAILED");
		Emit("=======================================================");
	}
}
