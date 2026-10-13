// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#include "global.h"	// include first in any cpp
#include "DeferredRelease.h"

DeferredReleaseManager g_deferredRelease;

void DeferredReleaseManager::Release(const utils::Buffer& data)
{
	if (data.buffer == VK_NULL_HANDLE && data.allocation == nullptr)
		return;

	m_pending.push_back({ data, m_currentFrameValue });
}

void DeferredReleaseManager::Update(utils::ResourceAllocator& allocator, uint64 completedValue)
{
	size_t writeIndex = 0;

	for (size_t readIndex = 0; readIndex < m_pending.size(); ++readIndex)
	{
		if (m_pending[readIndex].retireAtFrame <= completedValue)
			allocator.destroyBuffer(m_pending[readIndex].data);
		else
			m_pending[writeIndex++] = m_pending[readIndex];
	}

	m_pending.resize(writeIndex);
}

void DeferredReleaseManager::deinit(utils::ResourceAllocator& allocator)
{
	for (Entry& entry : m_pending)
		allocator.destroyBuffer(entry.data);

	m_pending.clear();
}
