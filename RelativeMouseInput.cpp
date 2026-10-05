#include "RelativeMouseInput.h"

CRelativeMouseInput::CRelativeMouseInput()
	: UserCapture(0), UserData(0), ObservedWindow(0), bCaptured(false)
{
}

CRelativeMouseInput::~CRelativeMouseInput()
{
	Current.ButtonMask = MIB_None;
	UpdateCaptureState();
}

bool CRelativeMouseInput::IsCaptured(const char* User) const
{
	if (User)
	{
		return UserCapture == User;
	}
	else
	{
		return UserCapture != 0;
	}
}

void CRelativeMouseInput::Create(HWND Window)
{
	ASSERT(Window);

	ObservedWindow = Window;

	// Subscribe to Raw Input for the mouse HID collection (usage page 1, usage 2). Raw Input
	// reports MOUSE_MOVE_ABSOLUTE/relative directly from the driver, which is what lets WndProc()
	// tell a real mouse apart from an absolute pointer (Wacom tablet, touch, RDP) reliably --
	// unlike guessing from how a WM_MOUSEMOVE message was synthesized, which some tablet drivers
	// don't tag consistently. dwFlags=0 delivers WM_INPUT only while this window has focus, which
	// matches the existing capture semantics (capturing implies the window is active).
	RAWINPUTDEVICE rid{};
	rid.usUsagePage = 0x01;	// HID_USAGE_PAGE_GENERIC
	rid.usUsage = 0x02;		// HID_USAGE_GENERIC_MOUSE
	rid.dwFlags = 0;
	rid.hwndTarget = Window;

	const BOOL ok = RegisterRawInputDevices(&rid, 1, sizeof(rid));
	ASSERT(ok);
}

void CRelativeMouseInput::WndProc(HWND Wnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (Wnd != ObservedWindow)
	{
		// Create missing or too early
		return;
	}

	// needed for glfw, otherwise we get relative mouse movement where there isn't any
	if (message == WM_CAPTURECHANGED)
		return;

	switch (message)
	{
	case WM_LBUTTONUP:
	case WM_LBUTTONDOWN:
	case WM_RBUTTONUP:
	case WM_RBUTTONDOWN:
	case WM_MBUTTONUP:
	case WM_MBUTTONDOWN:
	case WM_MOUSEMOVE:
	{
		Current.ButtonMask = (MouseInputButtons)(wParam & MIB_All);
		//			UpdateCaptureState();

	//				glm::ivec2 GlobalPos(LOWORD(lParam), HIWORD(lParam));
	}
	break;

	case WM_INPUT:
		if (IsCaptured())
			HandleRawInput((HRAWINPUT)lParam);
		break;
	}

	if (!IsCaptured())
	{
		GetCursorPos((POINT*)&Current.AbsolutePos);
	}
}

void CRelativeMouseInput::HandleRawInput(HRAWINPUT hRawInput)
{
	BYTE buffer[sizeof(RAWINPUT)];
	UINT size = sizeof(buffer);

	if (GetRawInputData(hRawInput, RID_INPUT, buffer, &size, sizeof(RAWINPUTHEADER)) == (UINT)-1)
		return;

	const RAWINPUT* raw = (const RAWINPUT*)buffer;

	if (raw->header.dwType != RIM_TYPEMOUSE)
		return;

	const RAWMOUSE& mouse = raw->data.mouse;

	if (mouse.usFlags & MOUSE_MOVE_ABSOLUTE)
	{
		// Pen/touch/RDP: lLastX/Y are normalized 0..65535 coordinates over either the virtual
		// desktop (MOUSE_VIRTUAL_DESKTOP) or the primary monitor -- convert to real screen pixels
		// and diff against the last sample ourselves, since the device gives us position, not delta.
		const bool bVirtualDesktop = (mouse.usFlags & MOUSE_VIRTUAL_DESKTOP) != 0;

		const int originX = bVirtualDesktop ? GetSystemMetrics(SM_XVIRTUALSCREEN) : 0;
		const int originY = bVirtualDesktop ? GetSystemMetrics(SM_YVIRTUALSCREEN) : 0;
		const int width = bVirtualDesktop ? GetSystemMetrics(SM_CXVIRTUALSCREEN) : GetSystemMetrics(SM_CXSCREEN);
		const int height = bVirtualDesktop ? GetSystemMetrics(SM_CYVIRTUALSCREEN) : GetSystemMetrics(SM_CYSCREEN);

		const glm::ivec2 screenPos(
			originX + MulDiv((int)mouse.lLastX, width, 65535),
			originY + MulDiv((int)mouse.lLastY, height, 65535));

		if (bHasLastAbsRawPos)
		{
			Current.RelativeX += screenPos.x - LastAbsRawPos.x;
			Current.RelativeY += screenPos.y - LastAbsRawPos.y;
		}
		// else: first absolute sample since capture began -- no prior reference yet, so this one
		// only establishes LastAbsRawPos rather than producing a delta (which would otherwise be
		// a jump from wherever the cursor happened to be to the pen's true position).

		LastAbsRawPos = screenPos;
		bHasLastAbsRawPos = true;
	}
	else
	{
		// Real relative mouse: lLastX/Y are already a delta, straight from the driver -- no cursor
		// position math needed at all, so this is immune to screen edges/cursor warping entirely.
		Current.RelativeX += mouse.lLastX;
		Current.RelativeY += mouse.lLastY;
	}
}

bool CRelativeMouseInput::SetUserCapture(const char* User, void* Data)
{
	ASSERT(User);

	if (UserCapture)
	{
		return false;
	}

	UserCapture = User;
	UserData = Data;

	UpdateCaptureState();

	return true;
}

void CRelativeMouseInput::ResetUserCapture(const char* User)
{
	if (UserCapture != User)
		return;

	UserCapture = 0;
	UserData = 0;

	UpdateCaptureState();
}

void CRelativeMouseInput::UpdateCaptureState()
{
	bool bNewCaptureCaptureState = UserCapture != 0;

	if (bNewCaptureCaptureState == bCaptured)
	{
		return;
	}

	if (!UserCapture)
	{
		int f = 0;
	}

	bCaptured = bNewCaptureCaptureState;

	if (bNewCaptureCaptureState)
	{
		GetCursorPos((POINT*)&Current.AbsolutePos);

		ShowCursor(false);
		SetCapture(ObservedWindow);

		// No stale reference for the first absolute-pointer sample of this drag -- see
		// HandleRawInput(). A real mouse doesn't need any position bookkeeping to start with,
		// since Raw Input gives it true deltas regardless of where the cursor is.
		bHasLastAbsRawPos = false;

		/*
				RECT rect;
				GetWindowRect(ObservedWindow, &rect);
				ClipCursor(&rect);
		*/
	}
	else
	{
		// to disable WndProc
		HWND BackupObservedWindow = ObservedWindow;
		ObservedWindow = 0;

		ReleaseCapture();

		SetCursorPos(Current.AbsolutePos.x, Current.AbsolutePos.y);
		ShowCursor(true);
		Current.RelativeX = 0;
		Current.RelativeY = 0;

		ObservedWindow = BackupObservedWindow;
	}
}
