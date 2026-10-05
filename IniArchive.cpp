// copyright DearGPU

#include "global.h"
#include <vector>
#include "ASCIIFile.h" // IO_GetFileSize()
#include "IniArchive.h"
#include "math.h"

// 32bit max file size
// todo: better error handling (bool IsValid() so not every operation needs to be checked)
class CGeneralArchive {
public:
	CGeneralArchive() : bIsLoading(false), outHandle(0), LoadPos(0) {}

	~CGeneralArchive() {
		if (IsSaving()) {
			uint32 FileSize = (uint32)Data.size();

			if (fwrite(&Data[0], FileSize, 1, outHandle) != 1) {
				ASSERT(0);
			}
			fclose(outHandle);
		}
	}

	// @ param bZeroTermination allows to a byte of 0 termination, could be made working for unicode
	// as well
  //  bool OpenForLoading(const TCHAR* pFileName, bool bZeroTermination = false) {
	bool OpenForLoading(const char* pFileName, bool bZeroTermination = false)
	{
		ASSERT(pFileName);
		ASSERT(!IsLoading());
		ASSERT(!IsSaving());
		ASSERT(!Data.size());

		FILE* in = 0;

		//    if (_wfopen_s(&in, pFileName, L"rb") != 0) {
		if (fopen_s(&in, pFileName, "rb") != 0) {
			return false;
		}

		size_t FileSize = IO_GetFileSize(pFileName);

		if (FileSize == -1) {
			// internal error
			return false;
		}

		Data.resize(FileSize + bZeroTermination);

		if (FileSize) {
			if (fread(&Data[0], FileSize, 1, in) != 1) {
				ASSERT(0);
			}
		}
		if (bZeroTermination) {
			Data[FileSize] = 0;
		}

		bIsLoading = true;

		fclose(in);
		return true;
	}

	//  bool OpenForSaving(const TCHAR* pFileName) {
	bool OpenForSaving(const char* pFileName) {
		ASSERT(pFileName);
		ASSERT(!IsLoading());
		ASSERT(!IsSaving());
		ASSERT(!Data.size());

		// 1MB reserve to avoid reallocations
		Data.reserve(1024 * 1024);

		//    if (_wfopen_s(&outHandle, pFileName, L"wb") != 0) {
		if (fopen_s(&outHandle, pFileName, "wb") != 0) {
			return false;
		}

		ASSERT(IsSaving());
		return true;
	}

	bool IsLoading() const {
		return bIsLoading;
	}

	bool IsSuccessfulLoading() const {
		return IsLoading() && LoadPos > 0 && LoadPos == Data.size();
	}

	bool IsSaving() const {
		return outHandle != 0;
	}

	// @param InChunkId e.g. 'Chk8' or 'BVH_'
	bool Chunk(const uint32 InChunkId) {
		ASSERT(InChunkId);

		uint32 BackupPos = LoadPos;

		uint32 ChunkId = InChunkId;

		Serialize(&ChunkId, sizeof(ChunkId));

		if (IsSaving()) {
			return true;
		}

		if (ChunkId == InChunkId) {
			return true;
		}

		LoadPos = BackupPos;
		return false;
	}

	void Serialize(void* InData, uint32 InSize) {
		ASSERT(InData);
		ASSERT(InSize);

		if (IsLoading()) {
			uint32 MaxSize = (uint32)Data.size();
			ASSERT(LoadPos + InSize <= MaxSize);
			memcpy(InData, &Data[LoadPos], InSize);
			LoadPos += InSize;
		}
		else {
			uint32 OldSize = (uint32)Data.size();
			Data.resize(OldSize + InSize);
			memcpy(&Data[OldSize], InData, InSize);
		}
	}

	size_t GetSize() const {
		return Data.size();
	}

	const uint8* GetDataPtr() const {
		return &Data[0];
	}

public:
	bool bIsLoading;
	FILE* outHandle;

	std::vector<uint8> Data;
	uint32 LoadPos;
};

void ParseWhiteSpace(const uint8*& p) {
	while (*p != 0 && (uint8)*p <= ' ') {
		++p;
	}
}

bool ParseStartsWith(const uint8*& _p, const char* Token) {
	const uint8* p = _p;
	const uint8* t = (const uint8*)Token;

	while (*t) {
		if (*p != *t) {
			return false;
		}
		++p;
		++t;
	}

	_p = p;
	return true;
}

void ParseLine(const uint8*& p, std::string& Out) {
	Out.clear();

	// can be optimized, does a lot of resize
	while (*p) {
		if (*p == 13) // CR
		{
			++p;

			if (*p == 10) // CR+LF
				++p;

			break;
		}
		if (*p == 10) // LF
		{
			++p;
			break;
		}

		Out += *p++;
	}
}

// without Numbers
bool IsNameCharacter(uint8 Value) {
	return (Value >= 'a' && Value <= 'z') || (Value >= 'A' && Value <= 'Z') || Value == '_';
}

bool IsDigitCharacter(uint8 Value) {
	return Value >= '0' && Value <= '9';
}

bool isValidKey(const char* key)
{
	if (!key)
		return false;

	for (const char* p = key; *p; ++p)
	{
		if (!IsNameCharacter(*p) && !IsDigitCharacter(*p) && *p != '_')
			return false;
	}

	return true;
}

bool ParseName(const uint8*& p, std::string& Out) {
	bool Ret = false;

	Out.clear();

	// can be optimized, does a lot of resize

	if (IsNameCharacter(*p)) {
		Out += *p++;
		Ret = true;
	}

	while (IsNameCharacter(*p) || IsDigitCharacter(*p)) {
		Out += *p++;
	}

	return Ret;
}

//bool CIniArchive::Load(const TCHAR* pFileName)
bool CIniArchive::Load(const char* pFileName)
{
	ASSERT(pFileName);
	ASSERT(!bIsLoading);

	// includes 0 termination
	CGeneralArchive LoadedArchive;

	if (!LoadedArchive.OpenForLoading(pFileName, true)) {
		return false;
	}

	const uint8* pFile = LoadedArchive.GetDataPtr();

	std::string Key, Value;
	std::string Line;
	// lines separated by "\r\n"
	std::string PendingCommentBlock;

	while (*pFile) {
		ParseLine(pFile, Line);

		// parse line
		{
			const uint8* p = (const uint8*)Line.c_str();

			ParseWhiteSpace(p);
			if (!ParseStartsWith(p, ";")) {
				if (ParseName(p, Key)) {
					ParseWhiteSpace(p);

					if (ParseStartsWith(p, "=")) {
						ParseWhiteSpace(p);
						ParseLine(p, Value);

						SValueEx ValueEx;

						ValueEx.Value = Value;
						ValueEx.CommentBlock = PendingCommentBlock;
						PendingCommentBlock.clear();
						//						ValueEx.bLoadedFromFile = true;

						// duplicates override the former settings
						KeyValuePairs[Key] = ValueEx;
						continue;
					}
					else {
						// bad syntax in ini, ignoring line
						ASSERT(0);
					}
				}
			}
			else {
				if (!PendingCommentBlock.empty()) {
					PendingCommentBlock += "\r\n";
				}
				PendingCommentBlock += Line;
			}
		}
	}

	bIsLoading = true;

	return true;
}

bool StartsWithSeparatorLine(const char* CommentBlock) {
	// e.g. "; -------------------------\r\n"

	const uint8* p = (const uint8*)CommentBlock;

	ParseWhiteSpace(p);
	if (*p == ';') {
		++p;
		ParseWhiteSpace(p);
		if (*p == '-') {
			return true;
		}
	}

	return false;
}

//bool CIniArchive::Save(const TCHAR* pFileName) {
bool CIniArchive::Save(const char* pFileName) {
	ASSERT(pFileName);
	ASSERT(!bIsLoading);	// This means once instance for saving and one for loading? How can we maintain comments this way? Change?

	FILE* outHandle = 0;
	//  if (_wfopen_s(&outHandle, pFileName, L"wb") != 0) {
	if (fopen_s(&outHandle, pFileName, "wb") != 0) {
		return false;
	}

	bool bFirst = true;

	// append data that was missing
	for (auto it = KeyValuePairs.begin(), end = KeyValuePairs.end(); it != end; ++it) {
		if (!bFirst) {
			bool bStartsWithSeparatorLine = StartsWithSeparatorLine(it->second.CommentBlock.c_str());

			if (!bStartsWithSeparatorLine) {
				//				fprintf(outHandle, "; -------------------------\r\n");
			}
		}
		bFirst = false;

		if (!it->second.CommentBlock.empty()) {
			fprintf(outHandle, "%s\r\n", it->second.CommentBlock.c_str());
		}
		fprintf(outHandle, "%s = %s\r\n", it->first.c_str(), it->second.Value.c_str());
	}

	fclose(outHandle);

	return true;
}

CIniArchive::SValueEx* CIniArchive::GetByKey(std::string Key) {
	auto it = KeyValuePairs.find(Key);

	if (it != KeyValuePairs.end()) {
		return &(it->second);
	}

	return nullptr;
}

void CIniArchive::AddKeyValue(
	const char* Key,
	const std::string& Value,
	const char* InCommentBlock) {
	SValueEx ValueEx;

	ASSERT(!Value.empty());

	ValueEx.Value = Value;
	//	ValueEx.bLoadedFromFile = false;

	if (InCommentBlock) {
		ValueEx.CommentBlock = InCommentBlock;
	}

	auto dataPair = std::pair<std::string, SValueEx>(Key, ValueEx);

	// duplicate key, use the last one but warn in debug
	//	ASSERT(KeyValuePairs.find(dataPair) == KeyValuePairs.end());

	KeyValuePairs.insert(dataPair);
}

void CIniArchive::AddKeyValue(const char* Key, float Value, const char* OptionalHelp)
{
	ASSERT(isValidKey(Key));

	char str[80];

	sprintf_s(str, sizeof(str) / sizeof(str[0]), "%g", Value);

	AddKeyValue(Key, str, OptionalHelp);
}

void CIniArchive::AddKeyValue(const char* Key, float3& Value, const char* OptionalHelp)
{
	ASSERT(isValidKey(Key));

	char str[80];

	sprintf_s(str, sizeof(str) / sizeof(str[0]), "%g %g %g", Value.x, Value.y, Value.z);

	AddKeyValue(Key, str, OptionalHelp);
}

void CIniArchive::AddKeyValue(const char* Key, int Value, const char* OptionalHelp) {
	ASSERT(isValidKey(Key));

	char str[80];

	sprintf_s(str, sizeof(str) / sizeof(str[0]), "%i", Value);

	AddKeyValue(Key, str, OptionalHelp);
}

// See Reflection.h/.cpp for the Reflection() free functions -- moved out of this file so
// per-class Reflection() overloads (e.g. CCamera's) have one shared home instead of piling up here.
