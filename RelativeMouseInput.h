#pragma once

#include "global.h"	// ASSERT
#include <glm/glm.hpp>
#include <glm/vec2.hpp> // glm::ivec2
#include "windows.h" // HWND

//
// 03/24/2010
// Martin Mittring
//
// Dependencies:
// * none
//
// streamlined for Win32
// can be adjusted to fit MFC as well
// handles absolute pointers (Wacom tablets, touch, RDP) as well as a relative mouse via
// Raw Input (see HandleRawInput() in the .cpp) -- Raw Input reports whether a sample is
// relative or absolute directly from the HID mouse driver, which is more reliable than
// guessing from how a WM_MOUSEMOVE message was synthesized (e.g. GetMessageExtraInfo()),
// since some tablet drivers don't route through Windows' Pointer Input Stack at all.

class CRelativeMouseInput
{
public:
	// same numbers as windows MB_...
	enum MouseInputButtons
	{
		MIB_LButton = 0x01,
		MIB_RButton = 0x02,
		MIB_Shift = 0x04,
		MIB_Control = 0x08,
		MIB_MButton = 0x10,
		//
		MIB_None = 0x00,
		MIB_All = 0x1f,
		MIB_InValid = -1,
	};

	struct MouseInputData
	{
		MouseInputData() :RelativeX(0), RelativeY(0), ButtonMask(MIB_InValid)
		{
			AbsolutePos.x = 0;
			AbsolutePos.y = 0;
		}

		bool IsValid() const
		{
			return ButtonMask != MIB_InValid;
		}

		// relative mouse movement, 0 when not captured
		int					RelativeX;
		// relative mouse movement, 0 when not captured
		int					RelativeY;
		// absolute mouse position on the screen (does not change when captured)
		glm::ivec2			AbsolutePos;
		// current button state
		MouseInputButtons	ButtonMask;
	};

	// -------------------------------------------------------

	// constructor
	CRelativeMouseInput();//MouseInputButtons InCaptureMask = MIB_RButton);
	// destructor
	~CRelativeMouseInput();

	void Create(HWND Wnd);

	// call before your windows message loop
	void WndProc(HWND Wnd, UINT message, WPARAM wParam, LPARAM lParam);

	const MouseInputButtons GetMouseButtons() const
	{
		return Current.IsValid() ? Current.ButtonMask : MouseInputButtons(0);
	}

	// @param User must not be 0
	const MouseInputData ClaimMouseInputData(const char* User, void* Data = 0)
	{
		ASSERT(User);

		MouseInputData Ret = Current;

		if (IsCaptured(User) && UserData == Data)
		{
			ResetRelativeMovement();
			return Ret;
		}

		return MouseInputData();
	}

	// @param user 0: if capture at all, !=0 if capture by this user
	bool IsCaptured(const char* User = 0) const;

	// optional the mouse can be captured by calling this function
	// @param User must not be 0
	bool SetUserCapture(const char* User, void* Data = 0);

	// @param User must not be 0
	void ResetUserCapture(const char* User);

	void* UserData;

	glm::ivec2 GetAbsPos() const { return Current.AbsolutePos; }

private: // ---------------------------------------

	// capture if either user or Capture Mask is true
	void UpdateCaptureState();

	void ResetRelativeMovement()
	{
		Current.RelativeX = 0;
		Current.RelativeY = 0;
	}

	// handles a WM_INPUT message while captured: accumulates Current.RelativeX/Y from either a
	// true relative delta (mouse) or the difference from the last absolute sample (pen/touch/RDP)
	void HandleRawInput(HRAWINPUT hRawInput);

	// --------------------------------------------

	// current button state and position
	MouseInputData Current;
	// last absolute screen position seen from an absolute-pointer (pen/touch) Raw Input sample,
	// used to compute its next delta; only meaningful when bHasLastAbsRawPos is true
	glm::ivec2 LastAbsRawPos = glm::ivec2(0, 0);
	// false right after capture starts, until the first absolute-pointer sample arrives -- avoids
	// diffing against a stale/unrelated LastAbsRawPos and producing a bogus jump
	bool bHasLastAbsRawPos = false;
	// 0 if not, name is used for debugging
	const char* UserCapture;

	HWND ObservedWindow;

	bool bCaptured;
};
