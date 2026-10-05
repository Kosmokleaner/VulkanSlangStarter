#pragma once
#include "ObjManager.h"


class CameraComponent
{
public:

	// vertical field of view, in degrees -- see GetFieldOfView()/SetFieldOfView()
	float fieldOfView = 90.0f;
};


extern CObjManager<CCameraHandle, CameraComponent> g_CameraManager;
