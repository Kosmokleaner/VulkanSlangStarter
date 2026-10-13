// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "vulkanBase.h" // VkDevice, utils::ResourceAllocator

enum EEventFlags
{
	// has to be one of those:

	// e.g. renderer or locally
	EF_Type_Unknown = 0x0,
	// for shader hot reload
	EF_Type_Shader = 0x1,
	// e.g. index, vertex buffer, shader code
	EF_Type_Buffer = 0x2,
	//
	EF_Type_Texture = 0x3,
	//
	EF_Type_Mask = 0xff,

	// more flags
};

struct EventContext
{
	VkDevice device;
	utils::ResourceAllocator& allocator;
};

struct IAppInit
{
	IAppInit(EEventFlags flags);	// auto register in g_EventManager (todo: use linked list as std::vector might not be ready)
	virtual ~IAppInit();
	virtual void init(EventContext &context) = 0;
	virtual void deinit(EventContext &context) = 0;
	virtual void reload(EventContext &context) {}
};


class EventManager
{
	struct Entry
	{
		// must not be 0
		IAppInit* ptr = {};
		EEventFlags flags;
	};

	// must not include 0
	std::vector<Entry> m_init;
public:

	// @param must not be 0
	void addAppInit(IAppInit* what, EEventFlags flags);
	// @param must not be 0
	void subAppInit(IAppInit* what);

	// todo: fix undefined order (defined within a c++ file is used as globals but not across files)

	// todo: consider struct e.g. VulkanContext
	void callAppInit(EEventFlags type, EventContext &context);
	// todo: consider struct e.g. VulkanContext
	// @param device must not be 0
	void callAppDeInit(EEventFlags type, EventContext &context);
	//
	void callReloadShaders(EventContext& context);
};

extern EventManager g_EventManager;
