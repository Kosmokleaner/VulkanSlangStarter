#include "global.h"	// include first in any cpp
#include "Scene.h"
#include "AppendArena.h"
#include "Logger.h"


Scene g_scene;

void Scene::save(TokenAppendArena &arena)
{
	for(uint32 i = 0, count = m_sceneObjects.GetElementCount(); i < count; ++i)
	{
		// todo: could be optimized to single function call
		auto hnd = m_sceneObjects.GetHandleByIterator(i);
		auto &el = m_sceneObjects.GetElementByIterator(i);

		TypedHandle dstKey(hnd, 0);	// 0 is Entity

		LOGI_AA_SAVE("Entity[%d] RenderComp=%d", dstKey.Index, el.hndRenderComp.Index);

		// todo: use reflection
		{
			ATokenCmdNew cmd;
			cmd.dstKey = dstKey;
			arena.appendCmd(cmd);
		}
		{
			ATokenCmdBlob blob(sizeof(Transform));
			arena.appendCmd(blob);

			uint64 valueP = arena.appendObject(el.transform);

			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 0;
			cmd.valueOrIndexInArenaToValue = valueP;
			arena.appendCmd(cmd);
		}
		{
			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 1;
			cmd.valueOrIndexInArenaToValue = el.hndRenderComp.getRaw();
			arena.appendCmd(cmd);
			cmd.memberIndex = 2;
			cmd.valueOrIndexInArenaToValue = el.hndLight.getRaw();
			arena.appendCmd(cmd);
			cmd.memberIndex = 3;
			cmd.valueOrIndexInArenaToValue = el.hndCamera.getRaw();
			arena.appendCmd(cmd);
		}
	}
}

void load(Entity& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping)
{
	// todo: reflection with backward compatible loading
	     if (cmd->memberIndex == 0) get(cmd, inoutRange, ref.transform);
	else if (cmd->memberIndex == 1) getHandle(cmd, remapping, 3, ref.hndRenderComp);
	else if (cmd->memberIndex == 2) getHandle(cmd, remapping, 4, ref.hndLight);
	else if (cmd->memberIndex == 3) getHandle(cmd, remapping, 5, ref.hndCamera);
	else ASSERT(0);
}
