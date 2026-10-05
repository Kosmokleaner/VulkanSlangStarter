#include "global.h"
#include "ShaderCache.h"
#include "ASCIIFile.h"
#include "Parse.h"

CShaderCache g_ShaderCache;


bool CShaderCache::Load()
{
	FILE *in = 0;

	if(_wfopen_s(&in, L"Cache/ShaderCache.dat", L"rb") != 0)
	{
		return false;
	}

	uint32 Magic = 0;
	if(fread(&Magic, sizeof(Magic), 1, in) != 1)
	{
		// make shadercache.dat write enable
		ASSERT(0);
		return false;
	}

	uint32 Version = 0;
	if(fread(&Version, sizeof(Version), 1, in) != 1)
	{
		ASSERT(0);
		return false;
	}

	uint32 Count = 0;
	if(fread(&Count, sizeof(Count), 1, in) != 1)
	{
		ASSERT(0);
		return false;
	}

	if(Magic != 'Shad')
	{
		ASSERT(0);
		return false;
	}

	if(Version != 2)
	{
		// if version gets updated we silently create a new cache from scratch
		return false;
	}

	for(uint32 i = 0; i < Count; ++i)
	{
		std::pair<CShaderHash, std::vector<uint32> > entry;

		if(fread(&entry.first, sizeof(entry.first), 1, in) != 1)
		{
			ASSERT(0);
			return false;
		}

		uint32 Size = 0;
		if(fread(&Size, sizeof(Size), 1, in) != 1)
		{
			ASSERT(0);
			return false;
		}

		entry.second.resize(Size);

		if(fread(&entry.second[0], Size * sizeof(uint32), 1, in) != 1)
		{
			ASSERT(0);
			return false;
		}
		
		g_ShaderCache.KeyValuePairs.insert(entry);
	}

	fclose(in);

	return true;
}

bool CShaderCache::Save()
{
	FILE *out = 0;

	if(_wfopen_s(&out, L"Cache/ShaderCache.dat", L"wb") != 0)
	{
		ASSERT(0);
		return false;
	}

	uint32 Magic = 'Shad';

	if(fwrite(&Magic, sizeof(Magic), 1, out) != 1)
	{
		ASSERT(0);
		return false;
	}
	uint32 Version = 2;

	if(fwrite(&Version, sizeof(Version), 1, out) != 1)
	{
		ASSERT(0);
		return false;
	}

	uint32 Count = (uint32)KeyValuePairs.size();

	if(fwrite(&Count, sizeof(Count), 1, out) != 1)
	{
		ASSERT(0);
		return false;
	}

	std::map<CShaderHash, std::vector<uint32> >::iterator it = KeyValuePairs.begin(), end = KeyValuePairs.end();
	for(; it != end; ++it)
	{
		const CShaderHash& key = it->first;
		const std::vector<uint32>& value = it->second;

		if(fwrite(&key, sizeof(key), 1, out) != 1)
		{
			ASSERT(0);
			return false;
		}

		uint32 Size = (uint32)value.size();
		if(fwrite(&Size, sizeof(Size), 1, out) != 1)
		{
			ASSERT(0);
			return false;
		}

		if(fwrite(&value[0], Size * sizeof(uint32), 1, out) != 1)
		{
			ASSERT(0);
			return false;
		}
	}

	fclose(out);

	return true;
}
	
// ------------------------------------------------------------------

// until any non path character
std::string ParsePath(const uint8* &p)
{
	std::string Ret;

	// to avoid reallocations
	Ret.reserve(32);

	for(;;)
	{
		uint8 c = *p++;

		if( (c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == '\\' ||
			c == '/' ||
			c == ' ' ||
			c == '@' ||
			c == '.' ||
			c == '_')
		{
			Ret += c;
		}
		else
		{
			break;
		}
	}

	return Ret;
}

// maintains "/" or "\"
static void appendPath(std::string &inoutPath, const std::string& pathWithFile)
{
	const char* lastSlash = nullptr;
	const char* pStart = pathWithFile.c_str();
	const char* p = pStart;
	while(*p)
	{
		if(*p == '\\' || *p == '/')
			lastSlash = p;
		++p;
	}

	if(lastSlash)
	{
		size_t oldSize = inoutPath.size();
		inoutPath.resize(oldSize + (lastSlash - pStart) + 1);
		memcpy(&inoutPath[oldSize], pStart, lastSlash - pStart + 1);

	}
}



// can be optimzized for less memory allocations
// @param parentPath e.g. "shaders/"
void ComputeShaderHashFromTxt(const char* FileTxt, std::string parentPath, /*ID3DInclude& Include, */CShaderHash& Hash)
{
	Hash.Add(FileTxt, (uint32)strlen(FileTxt));


	const uint8* p = (const uint8*)FileTxt;

	while(*p)
	{
		parseWhiteSpaceOrLF(p);

		// stupid simple, does not take C++ comments into account and can create false positives
		if(parseStartsWith(p, "#"))
		{
			parseWhiteSpaceOrLF(p);

			if(parseStartsWith(p, "include"))
			{
				parseWhiteSpaceOrLF(p);

				CASCIIFile file;
				if(parseStartsWith(p, "<"))
				{
					std::string path = ParsePath(p);
					std::string nextParentPath = parentPath;

					file.IO_LoadASCIIFile((parentPath + path).c_str());
					const char* ppData = file.GetDataPtr();

					if (ppData)// if 0 it might be a commented out include
					{
						appendPath(nextParentPath, path);

						ASSERT(ppData);
						ComputeShaderHashFromTxt((const char*)ppData, nextParentPath, /*Include, */Hash);		// todo: add non file path of path
					}
				}
				if(parseStartsWith(p, "\""))
				{
					std::string path = ParsePath(p);
					std::string nextParentPath = parentPath;

					file.IO_LoadASCIIFile((parentPath + path).c_str());
					const char* ppData = file.GetDataPtr();

					if(ppData)// if 0 it might be a commented out include
					{
						appendPath(nextParentPath, path);

						ASSERT(ppData);
						ComputeShaderHashFromTxt((const char*)ppData, nextParentPath, /*Include, */ Hash);		// todo: add non file path of path
					}
				}
			}
		}
		if(*p)++p;
	}
}


CShaderHash ComputeShaderHash(const char* FileName/*, ID3DInclude& Include*/)
{
	CASCIIFile file;

	file.IO_LoadASCIIFile(FileName);
	const char* txt = file.GetDataPtr();

	ASSERT(txt);

	CShaderHash Hash;
	ComputeShaderHashFromTxt(txt, std::string("shaders/"), /*Include, */Hash);

	return Hash;
}