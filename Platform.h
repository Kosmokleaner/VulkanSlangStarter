// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#pragma once

// useful
// call early in main, works with #pragma comment(linker, "/subsystem:windows")
// started from cmd or Powershell this make printf() work, even for #pragma comment(linker, "/subsystem:windows")
// @return there is a console
bool AttachParentConsole();

// Called by global.h's redefined ASSERT() macro in place of the CRT's own _wassert() when
// g_suppressAssertDialogs (App.cpp) is true: logs to stderr and terminates directly,
// with no dialog-capable code anywhere in the path (unlike _wassert(), which can show one
// regardless of report-mode settings that are supposed to prevent it).
void AppAssertFailed(const char* expression, const char* file, int line);

/*
// call early in main
// printf() and std::cout should go to a text console window
// open new console, even if started from console, printf go there, 
// without this the printf output is hidden
void OpenDebugConsole();
*/
