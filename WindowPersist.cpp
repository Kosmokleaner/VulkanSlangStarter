#include <windows.h>
#include "WindowPersist.h"
//#include <ImGui/GLFW/glfw3.h> // Will drag system OpenGL headers

// see https://www.glfw.org/docs/3.3/window_guide.html

bool WindowPersist::IsValid() const
{
	return data[2] > 0 && data[3] > 0;
}

#if WINDOW_PERSIST_HWND == 1
void WindowPersist::ApplyState(HWND hnd) const
{
	WINDOWPLACEMENT placement = {};
	placement.length = sizeof(WINDOWPLACEMENT);
	placement.rcNormalPosition.left = data[0];
	placement.rcNormalPosition.top = data[1];
	placement.rcNormalPosition.right = data[0] + data[2];
	placement.rcNormalPosition.bottom = data[1] + data[3];
	//	if(maximized)
	//	placement.flags = WPF_RESTORETOMAXIMIZED;
	placement.showCmd = (maximized ? SW_MAXIMIZE : 0) | SW_SHOW;
	//todo	placement.showCmd = maximized ? SW_MAXIMIZE : 0;

	//todo	SetWindowPlacement(hnd, &placement);
	//	SetWindowPos(hnd, 0, data[0], data[1], data[2], data[3], 
	//		maximized ? SW_MAXIMIZE : 0);
}

void WindowPersist::SaveState(HWND hnd)
{
	*this = {};

	WINDOWPLACEMENT placement = {};
	placement.length = sizeof(WINDOWPLACEMENT);
	GetWindowPlacement(hnd, &placement);
	data[0] = placement.rcNormalPosition.left;
	data[1] = placement.rcNormalPosition.top;
	data[2] = placement.rcNormalPosition.right - data[0];
	data[3] = placement.rcNormalPosition.bottom - data[1];
	maximized = fullscreen = (placement.flags & WPF_RESTORETOMAXIMIZED) != 0;

	/*
	maximized = fullscreen = IsZoomed(hnd);

	RECT rect;
	if(GetWindowRect(hnd, &rect))
	{
		data[0] = rect.left;
		data[1] = rect.top;
		data[2] = rect.right  - rect.left;
		data[3] = rect.bottom - rect.top;
	}
*/
}

#else // WINDOW_PERSIST_HWND == 1

void WindowPersist::ApplyState(GLFWwindow* window) const
{
	if (!window)
		return;

	if (fullscreen || maximized)
		glfwMaximizeWindow(window);
	else
	{
		//		glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
		//		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
				// without this call the glfwSetWindowPos() call removes the window title bar
		//		glfwShowWindow(window);
		glfwSetWindowPos(window, data[0], data[1]);
		glfwSetWindowSize(window, data[2], data[3]);
	}
}

void WindowPersist::SaveState(GLFWwindow* window)
{
	if (!window)
		return;

	maximized = fullscreen = glfwGetWindowMonitor(window) != nullptr;
	glfwGetWindowPos(window, &data[0], &data[1]);
	glfwGetWindowSize(window, &data[2], &data[3]);
}
#endif // WINDOW_PERSIST_HWND == 1

