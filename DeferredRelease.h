// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vulkanBase.h"

// Frame-safe deferred destruction for GPU-visible buffers. Buffer::deinit() (Buffer.h/.cpp)
// enqueues here instead of destroying immediately, since the buffer may still be referenced by
// GPU work that hasn't finished executing yet (this frame's command buffer, not yet submitted
// or not yet complete) -- deinit()+init() on the same Buffer is safe to call every frame (e.g.
// rebuilding an acceleration structure each frame) with no bookkeeping at the call site, the way
// a DX11-style driver hides this.
//
// Usage (see App.cpp's render loop):
//   - BeginFrame(pendingSignalValue) once per frame, before recording starts, with the timeline
//     value this frame's submission will eventually signal.
//   - Update(allocator, completedValue) once per frame with a timeline value the GPU has
//     *proven* to have completed (e.g. right after waiting on it) -- frees everything queued
//     during a frame whose signal value has already passed.
//   - deinit(allocator) once at shutdown, after the device is confirmed idle, to free anything
//     still pending unconditionally.
class DeferredReleaseManager
{
public:
	void BeginFrame(uint64 pendingSignalValue) { m_currentFrameValue = pendingSignalValue; }

	// no-op for a buffer that was never actually created (e.g. deinit() on a never-init()'d Buffer)
	void Release(const utils::Buffer& data);

	void Update(utils::ResourceAllocator& allocator, uint64 completedValue);

	void deinit(utils::ResourceAllocator& allocator);

	// for tests/diagnostics only -- not needed by normal callers
	size_t GetPendingCount() const { return m_pending.size(); }

private:
	struct Entry
	{
		utils::Buffer data;
		// the timeline value that must be reached before `data` is safe to destroy
		uint64 retireAtFrame;
	};

	std::vector<Entry> m_pending;
	uint64 m_currentFrameValue = {};
};

extern DeferredReleaseManager g_deferredRelease;
