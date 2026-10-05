#pragma once

// Minimal, dependency-free test framework used by the engine's unsupervised test mode
// (see TestMode.h). Tests call TestReport::Check() with a human readable description;
// the report prints a running PASS/FAIL log plus a final summary. It never aborts the
// process (unlike ASSERT in extern/vk_framework.h), so one failing check does not stop
// the remaining tests from running and reporting.
//
// Output goes to stdout (visible when the exe is launched with output redirected, since
// the x64 build has no attached console) and, if SetOutputFile() is called, to a text
// file as well, so results are easy to find and diff even without a terminal.

#include <string>
#include <vector>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace tests
{
	// @return true if a and b are within eps of each other
	inline bool NearlyEqual(float a, float b, float eps = 0.0001f)
	{
		return fabsf(a - b) <= eps;
	}

	inline bool NearlyEqual(const float3& a, const float3& b, float eps = 0.0001f)
	{
		return NearlyEqual(a.x, b.x, eps) && NearlyEqual(a.y, b.y, eps) && NearlyEqual(a.z, b.z, eps);
	}

	inline bool NearlyEqual(const float4& a, const float4& b, float eps = 0.0001f)
	{
		return NearlyEqual(a.x, b.x, eps) && NearlyEqual(a.y, b.y, eps) && NearlyEqual(a.z, b.z, eps) && NearlyEqual(a.w, b.w, eps);
	}

	// quaternions q and -q represent the same rotation, so both signs are accepted
	inline bool NearlyEqual(const glm::quat& a, const glm::quat& b, float eps = 0.0001f)
	{
		float4 av(a.x, a.y, a.z, a.w);
		float4 bv(b.x, b.y, b.z, b.w);

		return NearlyEqual(av, bv, eps) || NearlyEqual(av, -bv, eps);
	}

	inline bool NearlyEqual(const glm::mat4& a, const glm::mat4& b, float eps = 0.0001f)
	{
		for (int col = 0; col < 4; ++col)
			if (!NearlyEqual(a[col], b[col], eps))
				return false;

		return true;
	}

	inline std::string ToString(const float3& v)
	{
		char buf[128];
		snprintf(buf, sizeof(buf), "(%.5f, %.5f, %.5f)", v.x, v.y, v.z);
		return buf;
	}

	inline std::string ToString(const glm::quat& q)
	{
		char buf[160];
		snprintf(buf, sizeof(buf), "quat(w=%.5f, x=%.5f, y=%.5f, z=%.5f)", q.w, q.x, q.y, q.z);
		return buf;
	}

	inline std::string ToString(const glm::mat4& m)
	{
		// printed row by row for readability; glm stores it column-major in memory,
		// this is only about how it looks in the log, not about memory layout
		char buf[512];
		snprintf(buf, sizeof(buf),
			"\n    | %8.4f %8.4f %8.4f %8.4f |\n    | %8.4f %8.4f %8.4f %8.4f |\n    | %8.4f %8.4f %8.4f %8.4f |\n    | %8.4f %8.4f %8.4f %8.4f |",
			m[0][0], m[1][0], m[2][0], m[3][0],
			m[0][1], m[1][1], m[2][1], m[3][1],
			m[0][2], m[1][2], m[2][2], m[3][2],
			m[0][3], m[1][3], m[2][3], m[3][3]);
		return buf;
	}

	struct TestResult
	{
		std::string suite;
		std::string name;
		bool passed = {};
		std::string detail;
	};

	// collects and prints results for one run of the test mode; not thread safe,
	// used single-threaded during startup before the render loop exists
	class TestReport
	{
	public:
		// @param name: short identifier, e.g. "quat_compose_order"
		// @param detail: human readable sentence describing what was checked and, on
		//   failure, expected vs actual -- this is also the documentation of behavior
		void Check(const std::string& suite, const std::string& name, bool condition, const std::string& detail);

		// optional: also mirror all output into a text file so results survive without a console
		void SetOutputFile(const std::string& path);

		void PrintSummary();

		int GetPassCount() const;
		int GetFailCount() const;
		bool AllPassed() const { return GetFailCount() == 0; }

	private:
		void Emit(const std::string& line);

		std::vector<TestResult> m_results;
		std::ofstream m_outFile;
	};
}
