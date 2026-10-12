#pragma once
#include <map>
#include <string>
#include <vector>
#include "hash-library/sha256.h"

class CShaderHash
{
public: 
	CShaderHash() {}

	SHA256 sha256;

	// for std::map<>
	bool operator<(const CShaderHash& rhs) const
	{
		CShaderHash* L = (CShaderHash*)this;
		CShaderHash* R = (CShaderHash*)&rhs;

		unsigned char bufferL[SHA256::HashBytes];
		unsigned char bufferR[SHA256::HashBytes];

		L->sha256.getHash(bufferL);
		R->sha256.getHash(bufferR);

		return memcmp(bufferL, bufferR, SHA256::HashBytes) < 0;
	}

	void Add(const char* Data, uint32 ByteCount)
	{
		sha256.add(Data, ByteCount);
	}
};


class CShaderCache
{
public:

	// ShaderHash([HLSLCode]) = [Bytecode]
	std::map<CShaderHash, std::vector<uint32> > KeyValuePairs;
	
	bool Load();
	bool Save();
};

CShaderHash ComputeShaderHash(const char* FileName/*, ID3DInclude& Include*/);

extern CShaderCache g_ShaderCache;