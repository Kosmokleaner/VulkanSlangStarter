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

// Sets a single member of ref (cmd->memberIndex says which) while loading a saved scene -- called
// from AppendArena.cpp's ATokenCmdSetMember::execute(), once per member, once per Entity. See
// AppendArena.h's top-of-file backward-compatibility comment before changing memberIndex mappings.
void load(Entity& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping);

