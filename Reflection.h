#pragma once
#include "IniArchive.h"
#include <glm/glm.hpp>

class CCamera;

// Reflects one named value to/from an ini archive: when archive.IsSaving(), the current
// inoutValue is written into archive; when archive.IsLoading(), inoutValue is overwritten from
// archive (falling back to its current value if the key isn't present) -- see CIniArchive.
void Reflection(CIniArchive& archive, const char* Key, float& inoutValue);
void Reflection(CIniArchive& archive, const char* Key, int& inoutValue);
// also works with bool Value
void Reflection(CIniArchive& archive, const char* Key, bool& inoutValue);
void Reflection(CIniArchive& archive, const char* Key, float3& inoutValue);

// Reflects every persisted CCamera member (position, rotation, field of view) as a single
// string value under Key, so multiple cameras can each get their own Key without their fields
// colliding in the ini's flat key namespace (unlike composing per-member keys, e.g. Key+"_Pos").
// CCamera keeps its members protected, so this goes through GetPos()/SetPos()/
// GetAbsRotation()/SetRotation()/GetFieldOfView()/SetFieldOfView() rather than needing friend
// access, unlike e.g. Reflection(CIniArchive&, WindowPersist&, App&) (App.cpp),
// which does need it.
void Reflection(CIniArchive& archive, const char* Key, CCamera& camera);
