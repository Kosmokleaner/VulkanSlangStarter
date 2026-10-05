#pragma once
#include "Transform.h"
#include "ObjManager.h"

class RenderComponent;
class TokenAppendArena;
struct ATokenCmdSetMember;
struct ArenaTokenRange;
class HandleRemapTable;

struct Entity
{
	//
	Transform transform;

	// components:
	
	//
	CRenderCompHandle hndRenderComp;
	//
	CLightHandle hndLight;
	//
	CCameraHandle hndCamera;

//
// uint32 renderMask;
// physics reference
// parent for transform hierarchy
// material override (or move from instance to there)
// reflection (array of members to iterate, local position, member name)
};

class Scene
{
public:

	CObjManager<CEntityHandle, Entity> m_sceneObjects;

	// append scene to arena
	void save(TokenAppendArena& arena);
};

extern Scene g_scene;
