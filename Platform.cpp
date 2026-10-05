#include "Platform.h"
#include "logger.h"	// LOGE -- writes to the attached console (like printf) and the log file both
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void VVerseAssertFailed(const char* expression, const char* file, int line)
{
	LOGE("Assertion failed: %s, file %s, line %d", expression, file, line);
	// Windows Terminal's ConPTY relays console output to the terminal asynchronously -- a process
	// that exits within milliseconds of writing (like this one, via abort() right below) can lose
	// that output to the race entirely, even though it reached the real console handle correctly
	// (confirmed live: the exact same message always lands in the log file). Flushing explicitly
	// and giving ConPTY a brief moment to catch up before the process actually dies is a known,
	// pragmatic mitigation for this -- it doesn't come up for a normal run (anything that does real
	// work for more than a few milliseconds before exiting has no trouble here).
	std::cout.flush();
	std::cerr.flush();
	Sleep(100);
	abort();
}

// main() is called, not WinMain() and a console is opened when starting not from console, printf goes to console, 
// bool hasConsole = GetConsoleWindow() != nullptr; is true
//#pragma comment(linker, "/subsystem:console")
// the alternative (default in project settings)
//#pragma comment(linker, "/subsystem:windows")

bool AttachParentConsole()
{
	if (!AttachConsole(ATTACH_PARENT_PROCESS))
		return false;

	FILE* f = 0;

	// Skip rebinding only when the caller explicitly redirected to a real file or pipe
	// (FILE_TYPE_DISK/FILE_TYPE_PIPE) -- unconditionally overwriting THAT with CONOUT$ would
	// silently discard `> file`/`| pipe` redirection. Every other case (FILE_TYPE_CHAR,
	// FILE_TYPE_UNKNOWN, ...) must still be rebound: a GUI-subsystem process's CRT-level stdout
	// can be unusable regardless of what raw handle type it happens to have inherited at startup
	// (confirmed live -- PowerShell/Windows Terminal handed this process an already-FILE_TYPE_CHAR
	// handle, which an earlier, stricter "only rebind if FILE_TYPE_UNKNOWN" version of this check
	// wrongly took as "already fine" and skipped, breaking even plain printf()).
	auto needsRebind = [](DWORD stdHandleId)
	{
		DWORD type = GetFileType(GetStdHandle(stdHandleId));
		return type != FILE_TYPE_DISK && type != FILE_TYPE_PIPE;
	};
	if (needsRebind(STD_OUTPUT_HANDLE))
		freopen_s(&f, "CONOUT$", "w", stdout);
	if (needsRebind(STD_ERROR_HANDLE))
		freopen_s(&f, "CONOUT$", "w", stderr);
	if (needsRebind(STD_INPUT_HANDLE))
		freopen_s(&f, "CONIN$", "r", stdin);
	fflush(stdout);
	fflush(stderr);

	// Logger (extern/logger.h) writes through std::cout/std::cerr, not the C stdio FILE* the
	// freopen_s calls above just repointed -- those C++ streams can still be latched onto
	// whatever (failed/no-op) state they had before the console was attached, so nothing they
	// write actually reaches the console without this. Same fix already noted in this file's own
	// (unused) OpenDebugConsole() reference below; just missing here.
	std::cout.clear();
	std::cerr.clear();
	std::cin.clear();

	return true;
}

/*
void OpenDebugConsole()
{
	AllocConsole();

	FILE* stream;
	freopen_s(&stream, "CONOUT$", "w", stdout);
	freopen_s(&stream, "CONOUT$", "w", stderr);
	freopen_s(&stream, "CONIN$", "r", stdin);

	std::cout.clear();
	std::cerr.clear();
	std::cin.clear();
}

{
	bool hasConsole0 = AttachParentConsole();
	bool hasConsole1 = GetConsoleWindow() != nullptr;
	bool hasConsole2 = false;
	{
		HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
		if (h != INVALID_HANDLE_VALUE && h != nullptr)
		{
			DWORD mode;
			hasConsole2 = GetConsoleMode(h, &mode);
		}
	}
}

*/
