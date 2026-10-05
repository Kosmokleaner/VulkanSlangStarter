#include "global.h"
#include "CameraComponent.h"
#include "AppendArena.h"
#include "Transform.h"	// asuint()
#include "Logger.h"

CObjManager<CCameraHandle, CameraComponent> g_CameraManager;

// Position/orientation come from the owning Entity's own Transform (already serialized by
// Scene::save()), not from here.
void saveCameras(TokenAppendArena& arena)
{
	for (uint32 i = 0, count = g_CameraManager.GetElementCount(); i < count; ++i)
	{
		auto hnd = g_CameraManager.GetHandleByIterator(i);
		auto& el = g_CameraManager.GetElementByIterator(i);

		TypedHandle dstKey(hnd, 5);	// 5 is Camera

		LOGI_AA_SAVE("Camera[%d]", dstKey.Index);

		{
			ATokenCmdNew cmd;
			cmd.dstKey = dstKey;
			arena.appendCmd(cmd);
		}
		{
			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 0;
			cmd.valueOrIndexInArenaToValue = asuint(el.fieldOfView);
			arena.appendCmd(cmd);
		}
	}
}

void load(CameraComponent& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping)
{
	if (cmd->memberIndex == 0) get(cmd, inoutRange, ref.fieldOfView);
	else ASSERT(0);
}
