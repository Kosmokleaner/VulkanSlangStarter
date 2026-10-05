#include "global.h"	// include first in any cpp
#include "Reflection.h"

void Reflection(CIniArchive& archive, const char* Key, float& inoutValue) {
	ASSERT(isValidKey(Key));
	if (archive.IsSaving())
		archive.AddKeyValue(Key, inoutValue);
	else {
		if (CIniArchive::SValueEx* p = archive.GetByKey(Key)) {
			inoutValue = p->GetFloat(inoutValue);
		}
	}
}

void Reflection(CIniArchive& archive, const char* Key, float3& inoutValue) {
	ASSERT(isValidKey(Key));
	if (archive.IsSaving())
		archive.AddKeyValue(Key, inoutValue);
	else {
		if (CIniArchive::SValueEx* p = archive.GetByKey(Key)) {
			inoutValue = p->GetVec3(inoutValue);
		}
	}
}

void Reflection(CIniArchive& archive, const char* Key, int& inoutValue) {
	ASSERT(isValidKey(Key));

	if (archive.IsSaving())
		archive.AddKeyValue(Key, inoutValue);
	else {
		if (CIniArchive::SValueEx* p = archive.GetByKey(Key)) {
			inoutValue = p->GetInt(inoutValue);
		}
	}
}

void Reflection(CIniArchive& archive, const char* Key, bool& inoutValue) {
	ASSERT(isValidKey(Key));

	if (archive.IsSaving())
		archive.AddKeyValue(Key, inoutValue);
	else {
		if (CIniArchive::SValueEx* p = archive.GetByKey(Key)) {
			inoutValue = p->GetInt(inoutValue) != 0;
		}
	}
}
