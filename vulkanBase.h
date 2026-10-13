// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0
// Portions derived from src/minimal_latest.cpp in https://github.com/nvpro-samples/vk_minimal_latest,
// modified by Martin Mittring.

#pragma once

//--- VMA implementation bootstrap ----------------------------------------------
// The Vulkan Memory Allocator header is "single-TU": exactly one .cpp must
// define VMA_IMPLEMENTATION before including vk_mem_alloc.h. App.cpp is that .cpp
// (it defines it before including this header); everywhere else this header
// brings in vk_mem_alloc.h for the types only.
//
// Volk must precede VMA so vk_mem_alloc.h sees the Vulkan types.
#include "volk.h"

// Included here (before vk_mem_alloc.h) specifically so VMA_LEAK_LOG_FORMAT below can use LOGE --
// vk_framework.h normally brings this in, but that happens after vk_mem_alloc.h is included, which
// is too late for a macro that gets expanded from inside vk_mem_alloc.h itself. logger.h has no
// Vulkan/volk dependency, so pulling it in this early is safe.
#include "logger.h"

// like printf()
void Log(const char* format, ...);

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

// Was Log() (OutputDebugStringA only -- invisible without an attached debugger). Routed through
// the file logger instead so "on quit" leak dumps show up in <exe-name>.txt without needing a
// debugger attached.
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    LOGE((format), __VA_ARGS__);                                                                                     \
  }
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#pragma warning(push)
#pragma warning(disable : 4100)  // Unreferenced formal parameter
#pragma warning(disable : 4189)  // Local variable is initialized but not referenced
#pragma warning(disable : 4127)  // Conditional expression is constant
#pragma warning(disable : 4324)  // Structure was padded due to alignment specifier
#pragma warning(disable : 4505)  // Unreferenced function with internal linkage has been removed
#include "vk_mem_alloc.h"
#pragma warning(pop)
#ifdef __clang__
#pragma clang diagnostic pop
#endif

//--- Framework -----------------------------------------------------------------
// Brings in volk again (include-guarded, no-op), VMA types, GLFW, ImGui,
// logger.h, debug_util.h, STL; ASSERT / VK_CHECK macros; and the utils::
// namespace (Buffer/Image structs, barrier + layout helpers, Context,
// Swapchain, FramePacer, ResourceAllocator, SamplerPool, RenderTarget, findFile).
#include "vk_framework.h"

// renderer specific, todo: move

//#include "Buffer.h"
class Buffer;
