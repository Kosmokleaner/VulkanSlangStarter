// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#include "global.h"
#include "EventManager.h"

EventManager g_EventManager;

IAppInit::IAppInit(EEventFlags flags)
{
	g_EventManager.addAppInit(this, flags);
}

IAppInit::~IAppInit()
{
	g_EventManager.subAppInit(this);
}

void EventManager::subAppInit(IAppInit* what)
{
	ASSERT(what);
	for (const auto& el : m_init)
	{
		if (el.ptr == what)
		{
			auto index = &el - m_init.data();
			m_init.erase(m_init.begin() + index);
			return;
		}
	}
}

void EventManager::addAppInit(IAppInit* what, EEventFlags flags)
{
	ASSERT(what);
	// todo check dup
	m_init.push_back({ what, flags });
}

void EventManager::callAppInit(EEventFlags type, EventContext &context)
{
	ASSERT((type & ~EF_Type_Mask) == 0);

	// m_init doubles as the registration list for callAppDeInit() later -- objects register once,
	// in their IAppInit constructor, for their whole lifetime (not just until init() runs). Clearing
	// it here used to unregister everything right after startup, so callAppDeInit() at shutdown had
	// nothing left to iterate and every registered Buffer/Shader leaked.
	for (auto el : m_init)
	{
		if ((el.flags & EF_Type_Mask) == type || type == EF_Type_Mask)
			el.ptr->init(context);
	}
}

void EventManager::callAppDeInit(EEventFlags type, EventContext &context)
{
	ASSERT(context.device);
	ASSERT((type & ~EF_Type_Mask) == 0);

	// deinit() implementations don't unregister themselves, so iterate a local copy (also safe
	// against any that do end up mutating m_init, e.g. via their own destructor).
	std::vector<Entry> toDeinit;
	toDeinit.swap(m_init);

	for (auto el : toDeinit)
	{
		if ((el.flags & EF_Type_Mask) == type || type == EF_Type_Mask)
			el.ptr->deinit(context);
	}
}

void EventManager::callReloadShaders(EventContext& context)
{
	ASSERT(context.device);

	for (auto el : m_init)
	{
		if ((el.flags & EF_Type_Mask) == EF_Type_Shader)
			el.ptr->reload(context);
	}
}
