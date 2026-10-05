#pragma once

#include <cstdlib>  // unix size_t 

//! 0 byte terminted char ASCII file
class CASCIIFile
{
public:
	CASCIIFile();

	CASCIIFile(CASCIIFile& a);

	virtual ~CASCIIFile();

	const CASCIIFile operator=(const CASCIIFile& a);

	// release old, don't make copy
	// @param ptr has not be 0 terminated, allocated with malloc
	// @param size without 0 termination
	void CoverThisData(const char* ptr, const size_t size);

	size_t GetDataSize() const;

	const char* GetDataPtr();

	void ReleaseData();

	static bool IO_GetAvailability(const char* pathname);

	//	bool IO_SaveASCIIFile( const wchar_t* pathname );
	bool IO_SaveASCIIFile(const char* pathname);

	//	bool IO_LoadASCIIFile( const wchar_t* pathname );
	bool IO_LoadASCIIFile(const char* pathname);

protected: // ------------------------------------------------------------------------

	const char* m_Data; //!< zero terminated text
	size_t m_Size; //!< including the zero termination (otherwise 0)
};

size_t IO_GetFileSize(const char* Name);