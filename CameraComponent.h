#pragma once
#include "ObjManager.h"

class TokenAppendArena;
struct ATokenCmdSetMember;
struct ArenaTokenRange;
class HandleRemapTable;

class CameraComponent
{
public:

	// vertical field of view, in degrees -- see GetFieldOfView()/SetFieldOfView()
	float fieldOfView = 90.0f;
};


extern CObjManager<CCameraHandle, CameraComponent> g_CameraManager;

void saveCameras(TokenAppendArena& arena);
// Sets a single member of ref (cmd->memberIndex says which) while loading a saved scene -- called
// from AppendArena.cpp's ATokenCmdSetMember::execute(), once per member, once per camera. See
// AppendArena.h's top-of-file backward-compatibility comment before changing memberIndex mappings.
void load(CameraComponent& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping);