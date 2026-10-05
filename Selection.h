#pragma once
#include "ObjManager.h"
#include <set>

class Selection
{
public:

	std::set<CEntityHandle> m_sceneObjects;
	std::set<CRenderCompHandle> m_meshes;
	std::set<CMatHandle> m_materials;
	std::set<CLightHandle> m_lights;
	std::set<CCameraHandle> m_cameras;

	void clear();
};

