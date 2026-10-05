#pragma once

// Documents (via executable tests) the actual matrix and quaternion conventions this
// engine's C++ side relies on: glm's column-major storage, column-vector M*v multiply
// order, and how Transform.h (SRT) composes rotations/positions. See TestMode.h for how
// this suite is invoked. This intentionally only covers the C++/glm side; the shader-side
// row_major / mul() convention (shaders/shader_io.h ROW_MAJOR) is a separate concern and
// not exercised here.

namespace tests { class TestReport; }

namespace tests
{
	void RunGlmMathTests(TestReport& report);
}
