#include "global.h"
#include <glm/glm.hpp>
#include "Material.h"
#include "AppendArena.h"
#include "Transform.h"
#include "Logger.h"

CObjManager<CMatHandle, Material> g_MaterialManager;

void saveMaterials(TokenAppendArena& arena)
{
	for (uint32 i = 0, count = g_MaterialManager.GetElementCount(); i < count; ++i)
	{
		// todo: could be optimized to single function call
		auto hnd = g_MaterialManager.GetHandleByIterator(i);
		auto& el = g_MaterialManager.GetElementByIterator(i);

		TypedHandle dstKey(hnd, 1);	// 1 is Material

		LOGI_AA_SAVE("Material[%d] name=%s", dstKey.Index, el.m_name.c_str());

		// todo: use reflection
		{
			ATokenCmdNew cmd;
			cmd.dstKey = dstKey;
			arena.appendCmd(cmd);
		}
		{
			// todo: improve API !
			ATokenCmdBlob blob(sizeof(float3) * 2 + (el.m_name.size() + 1));
			arena.appendCmd(blob);

			size_t oldSize = arena.size();

			uint64 value1 = arena.appendObject(el.m_albedoColor);
			uint64 value2 = arena.appendObject(el.m_emissiveColor);
			uint64 value0 = arena.appendString(el.m_name);

			size_t newSize = arena.size();
			ASSERT(newSize == oldSize + blob.count);

			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 0;
			cmd.valueOrIndexInArenaToValue = value0;
			arena.appendCmd(cmd);
			cmd.memberIndex = 1;
			cmd.valueOrIndexInArenaToValue = value1;
			arena.appendCmd(cmd);
			cmd.memberIndex = 2;
			cmd.valueOrIndexInArenaToValue = value2;
			arena.appendCmd(cmd);
		}
		{
			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 3;
			cmd.valueOrIndexInArenaToValue = asuint(el.m_roughness);
			arena.appendCmd(cmd);
			cmd.memberIndex = 4;
			cmd.valueOrIndexInArenaToValue = asuint(el.m_metallic);
			arena.appendCmd(cmd);
			cmd.memberIndex = 5;
			cmd.valueOrIndexInArenaToValue = asuint(el.m_albedoTextureFade);
			arena.appendCmd(cmd);
			cmd.memberIndex = 9;
			cmd.valueOrIndexInArenaToValue = asuint(el.m_specOcclusionFromAlbedo);
			arena.appendCmd(cmd);
			cmd.memberIndex = 11;
			cmd.valueOrIndexInArenaToValue = (uint32)el.m_type;
			arena.appendCmd(cmd);
			cmd.memberIndex = 12;
			cmd.valueOrIndexInArenaToValue = asuint(el.m_emissiveIntensity);
			arena.appendCmd(cmd);
		}
		{
			ATokenCmdSetMember cmd;
			cmd.dstKey = dstKey;
			cmd.memberIndex = 6;
			cmd.valueOrIndexInArenaToValue = el.m_hndTexture.getRaw();
			arena.appendCmd(cmd);
			cmd.memberIndex = 7;
			cmd.valueOrIndexInArenaToValue = el.m_hndMetallicRoughnessTexture.getRaw();
			arena.appendCmd(cmd);
			cmd.memberIndex = 8;
			cmd.valueOrIndexInArenaToValue = el.m_hndLightmap.getRaw();
			arena.appendCmd(cmd);
			cmd.memberIndex = 10;
			cmd.valueOrIndexInArenaToValue = el.m_hndNormalMap.getRaw();
			arena.appendCmd(cmd);
		}
	}
}

void load(Material& ref, ATokenCmdSetMember* cmd, const ArenaTokenRange& inoutRange, const HandleRemapTable& remapping)
{
	// todo: reflection with backward compatible loading
	     if (cmd->memberIndex == 0) get(cmd, inoutRange, ref.m_name);
	else if (cmd->memberIndex == 1) get(cmd, inoutRange, ref.m_albedoColor);
	else if (cmd->memberIndex == 2) get(cmd, inoutRange, ref.m_emissiveColor);
	else if (cmd->memberIndex == 3) get(cmd, inoutRange, ref.m_roughness);
	else if (cmd->memberIndex == 4) get(cmd, inoutRange, ref.m_metallic);
	else if (cmd->memberIndex == 5) get(cmd, inoutRange, ref.m_albedoTextureFade);
	else if (cmd->memberIndex == 6) getHandle(cmd, remapping, 2, ref.m_hndTexture);
	else if (cmd->memberIndex == 7) getHandle(cmd, remapping, 2, ref.m_hndMetallicRoughnessTexture);
	else if (cmd->memberIndex == 8) getHandle(cmd, remapping, 2, ref.m_hndLightmap);
	// Not present in an older save file (defaults to m_specOcclusionFromAlbedo's own default, 0 =
	// disabled) -- see Material.h's own comment.
	else if (cmd->memberIndex == 9) get(cmd, inoutRange, ref.m_specOcclusionFromAlbedo);
	else if (cmd->memberIndex == 10) getHandle(cmd, remapping, 2, ref.m_hndNormalMap);
	// Not present in an older save file (defaults to m_type's own default, EType::PBR).
	else if (cmd->memberIndex == 11)
	{
		uint32 v = 0;
		get(cmd, inoutRange, v);
		ref.m_type = (Material::EType)v;
	}
	// Not present in an older save file (defaults to m_emissiveIntensity's own default, 1.0 = off)
	// -- see Material.h's own comment.
	else if (cmd->memberIndex == 12) get(cmd, inoutRange, ref.m_emissiveIntensity);
	else ASSERT(0);
}