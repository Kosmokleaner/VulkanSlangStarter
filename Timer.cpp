// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#include "global.h"
#include "Timer.h"
#include <windows.h>

CTimer g_Timer;

CTimer::CTimer()
{
	QueryPerformanceFrequency((LARGE_INTEGER*)&m_PerformanceFrequency);
}


double CTimer::GetAbsoluteTime() const
{
	ASSERT(m_PerformanceFrequency);

	LARGE_INTEGER b;

	QueryPerformanceCounter(&b);

	return (double)(b.QuadPart) / (double)m_PerformanceFrequency;
}


CScopedCPUTimer::~CScopedCPUTimer()
{
	double EndTime = g_Timer.GetAbsoluteTime();

	float TimeDelta = (float)(EndTime - StartTime);

	/* hidden for now
		char str[1024];

		sprintf_s(str, sizeof(str), "CPU  %*.1f ms %s", 6, TimeDelta * 1000, Text);		// 6 spaces (see trick here: http://www.cplusplus.com/reference/cstdio/printf/)
		g_Renderer.textContext.Print(20, g_Renderer.textContext.CursorY, str);

		const int yStep = 11;
		g_Renderer.textContext.CursorY += yStep;
	*/

	/*
	if (!Name.empty())
	{
	OutputDebugStringA(": ");
	OutputDebugStringW(Name.c_str());
	}

	OutputDebugStringA("\n");
	*/
}


CScopedCPUTimerLog::~CScopedCPUTimerLog()
{
	double EndTime = g_Timer.GetAbsoluteTime();

	float TimeDelta = (float)(EndTime - StartTime);

	char str[1024];

	sprintf_s(str, sizeof(str) / sizeof(str[0]), " %*.1f ms ", 6, TimeDelta * 1000);		// 6 spaces (see trick here: http://www.cplusplus.com/reference/cstdio/printf/)
	OutputDebugStringA(str);

	sprintf_s(str, sizeof(str) / sizeof(str[0]), "  SCOPED_CPU_TIMER_LOG %s", Text);
	OutputDebugStringA(str);

	if (!Name.empty())
	{
		OutputDebugStringA(": ");
		OutputDebugStringW(Name.c_str());
	}

	OutputDebugStringA("\n");
}