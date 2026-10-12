

#pragma once

//==========================================

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

#include <glm/glm.hpp>

// HLSL/Slang-style aliases for glm's vector types -- still glm underneath, just spelled the way
// the shader code (and shader_io.h's shared C++/Slang structs) already spells them.
typedef glm::vec2 float2;
typedef glm::vec3 float3;
typedef glm::vec4 float4;

#define SLANG_DISABLE_EXCEPTIONS 1

#pragma warning (disable: 4858)     // discarding return value: This member function returns a bool indicating whether the collection is empty and has no other effects. It is not useful to call this member function and discard the return value. Use the 'clear()' member function if you want to erase all elements.
#pragma warning (disable: 4834)     // discarding return value of function with [[nodiscard]] attribute
#pragma warning (disable: 4530)     // warning C4530: C++ exception handler used, but unwind semantics are not enabled. Specify /EHsc

#include <cstdlib>	// std::abort(), used by ASSERT below
#include <intrin.h>	// __debugbreak(), ditto
#include "Platform.h"
#include "logger.h"

// Set true (main(), App.cpp, right after AttachParentConsole()) when a console launched
// this process -- see ASSERT's own comment just below for why it matters.
extern bool g_suppressAssertDialogs;
// Implemented in Platform.cpp -- logs and terminates
// directly, with no dialog-capable code anywhere in the path. See ASSERT's own comment.
void VVerseAssertFailed(const char* expression, const char* file, int line);

// ASSERT(condition) or ASSERT(condition, message) -- logs and aborts in release (NDEBUG), asserts
// in debug. The project's one and only assert mechanism: raw assert()/<assert.h> is NOT used
// anywhere in project code (only by vendored/extern libraries, which manage their own). Aborts
// rather than throwing: a broken invariant isn't safely recoverable via exception unwinding, so
// terminate directly instead. message is optional; when omitted, the stringified condition is
// used instead (matching plain assert()'s own behavior) -- and when given, it can be any
// const char* expression, not just a literal (VK_CHECK below passes a runtime-computed one).
//
// The debug branch checks g_suppressAssertDialogs directly instead of going through plain
// assert() -- <assert.h>'s own dialog (Abort/Retry/Ignore) has proven impossible to suppress
// through the documented APIs (_CrtSetReportHook, _CrtSetReportMode) even when confirmed to
// execute; VVerseAssertFailed() sidesteps that mechanism entirely instead of fighting it. This
// also avoids a second, structural problem: <assert.h> has no include guard (by design, so code
// can toggle NDEBUG mid-file), so any of it being re-included after a redefinition -- which
// vendored headers may do directly -- would silently undo that redefinition; checking the flag
// inline here has no such fragility.
#ifdef NDEBUG
#define VVERSE_ASSERT_IMPL(condition, message)                                                     \
	do                                                                                              \
	{                                                                                               \
		if (!(condition))                                                                           \
		{                                                                                            \
			LOGE("ASSERT failed at %s:%d: %s", __FILE__, __LINE__, message);                          \
			std::abort();                                                                              \
		}                                                                                            \
	} while (false)
#else
#define VVERSE_ASSERT_IMPL(condition, message)                                                     \
	do                                                                                              \
	{                                                                                               \
		if (!(condition))                                                                           \
		{                                                                                            \
			if (g_suppressAssertDialogs)                                                             \
				VVerseAssertFailed(message, __FILE__, __LINE__);                                      \
			else                                                                                     \
			{                                                                                        \
				/* Breaks directly into an attached debugger at this exact line (like a normal   */  \
				/* breakpoint) instead of routing through <assert.h>'s own dialog -- needs no CRT */  \
				/* assert machinery at all, so it isn't subject to the fragility documented above.*/ \
				LOGE("ASSERT failed at %s:%d: %s", __FILE__, __LINE__, message);                      \
				__debugbreak();                                                                       \
				std::abort();                                                                          \
			}                                                                                        \
		}                                                                                            \
	} while (false)
#endif

// Picks VVERSE_ASSERT2 (2 args given) or VVERSE_ASSERT1 (1 arg given) based on how many arguments
// ASSERT(...) was actually called with -- the standard "overload a macro by argument count" trick.
#define VVERSE_ASSERT_PICK(_1, _2, NAME, ...) NAME
#define ASSERT(...) VVERSE_ASSERT_PICK(__VA_ARGS__, VVERSE_ASSERT2, VVERSE_ASSERT1)(__VA_ARGS__)
#define VVERSE_ASSERT1(condition) VVERSE_ASSERT_IMPL(condition, #condition)
#define VVERSE_ASSERT2(condition, message) VVERSE_ASSERT_IMPL(condition, message)

#define MAX_PATH          260

// todo: move
inline uint32 short2Uint(uint32 h, uint32 l)
{
	ASSERT(h <= 0xffff);
	ASSERT(l <= 0xffff);

	return (h << 16) | l;
}

inline float asfloat(uint32 h)
{
	return *(float*)&h;
}

