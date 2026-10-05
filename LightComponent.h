#pragma once
#include "ObjManager.h"
#include <glm/glm.hpp>

class TokenAppendArena;
struct ATokenCmdSetMember;
struct ArenaTokenRange;
class HandleRemapTable;

class LightComponent
{
public:
	float m_lightLength = 6.0f;
	// 0..1
	float3 m_lightColor = float3(1.0f, 1.0f, 1.0f);
	// multiplier for m_lightColor
	float m_lightIntensity = 5.0f;

/*
	Reflection(archive, "LightLength", g_lightLength);
	Reflection(archive, "LightColor", g_lightColor);
	Reflection(archive, "LightIntensity", g_lightIntensity);
*/
};

extern CObjManager<CLightHandle, LightComponent> g_LightManager;

void saveLights(TokenAppendArena& arena);
// Sets a single member of ref (cmd->memberIndex says which) while loading a saved scene -- called
// from AppendArena.cpp's ATokenCmdSetMember::execute(), once per member, once per light. See
// AppendArena.h's top-of-file backward-compatibility comment before changing memberIndex mappings.
void load(LightComponent& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping);

