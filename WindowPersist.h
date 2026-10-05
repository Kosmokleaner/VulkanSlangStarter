#pragma once

#define WINDOW_PERSIST_HWND 1

class WindowPersist
{
public:

	bool fullscreen = false;
	bool maximized = false;
	// left, top, width, height
	int data[4] = { 100, 100, 1024, 768 };

#if WINDOW_PERSIST_HWND == 1

	void ApplyState(HWND hnd) const;
	// this changes the object internals
	// @param window 0 is silently ignored
	void SaveState(HWND hnd);

#else // WINDOW_PERSIST_HWND == 1

	// this changes the window
	// @param window 0 is silently ignored
	void ApplyState(struct GLFWwindow* window) const;
	// this changes the object internals
	// @param window 0 is silently ignored
	void SaveState(struct GLFWwindow* window);

#endif // WINDOW_PERSIST_HWND == 1

	bool IsValid() const;
};