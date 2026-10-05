#include "global.h"
#include "Selection.h"

void Selection::clear()
{
	m_sceneObjects.clear();
	m_meshes.clear();
	m_materials.clear();
	m_lights.clear();
	m_cameras.clear();
}
