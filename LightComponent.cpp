#include "global.h"
#include "LightComponent.h"
#include "imgui.h"
#include "AppendArena.h"
#include "Transform.h"	// asuint()
#include "Logger.h"

CObjManager<CLightHandle, LightComponent> g_LightManager;

// Position comes from the owning Entity's own Transform (already serialized by Scene::save()),
// not from here -- see updateLights() (App.cpp), which reads entity.transform.GetPos().
void saveLights(TokenAppendArena& arena)
{
	for (uint32 i = 0, count = g_LightManager.GetElementCount(); i < count; ++i)
	{
		auto hnd = g_LightManager.GetHandleByIterator(i);
		auto& el = g_LightManager.GetElementByIterator(i);

		TypedHandle dstKey(hnd, 4);	// 4 is Light

		LOGI_AA_SAVE("Light[%d]", dstKey.Index);

		{
			ATokenCmdNew cmd;
			cmd.dstKey = dstKey;
			arena.appendCmd(cmd);
		}
		{
			ATokenCmdBlob blob(sizeof(float3));
			arena.appendCmd(blob);

			size_t oldSize = arena.size();
			uint64 value1 = arena.appendObject(el.m_lightColor);
			size_t newSize = arena.size();
			ASSERT(newSize == oldSize + blob.count);

			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 0;
			cmd.valueOrIndexInArenaToValue = asuint(el.m_lightLength);
			arena.appendCmd(cmd);
			cmd.memberIndex = 1;
			cmd.valueOrIndexInArenaToValue = value1;
			arena.appendCmd(cmd);
			cmd.memberIndex = 2;
			cmd.valueOrIndexInArenaToValue = asuint(el.m_lightIntensity);
			arena.appendCmd(cmd);
		}
	}
}

void load(LightComponent& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping)
{
	     if (cmd->memberIndex == 0) get(cmd, inoutRange, ref.m_lightLength);
	else if (cmd->memberIndex == 1) get(cmd, inoutRange, ref.m_lightColor);
	else if (cmd->memberIndex == 2) get(cmd, inoutRange, ref.m_lightIntensity);
	else ASSERT(0);
}
