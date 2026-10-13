// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#include "global.h"
#include <stdlib.h>
#include <malloc.h>														// malloc,free
#include <stdio.h>														// FILE
#include <fcntl.h>														// filesize
#include <string.h>														// strlen

#ifdef _WIN32
#include <io.h> // filesize
#define NOMINMAX
#include <windows.h> // GetFileSize()
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "ASCIIFile.h"

#pragma warning( disable : 4996 )

/// test0

/// {} = This is a test {key2}
///    = And it goes further {key3}y as expected
/// {key3} = trick
/// {key2} = (Another {key3,separator})

/// {key3} = AAA
/// {key3} = BBB
/// {separator} = ;

//					++fbuf; 				/* Skip overlay byte */


// {} ="     Application: {AppName}"
//    ="  date(mm/dd/yy): {__Date__}"
//    ="  time(hh:mm:ss): {__Time__}"
//    ="      line count: {__CodeLines__}"
//    =" code file count: {__CodeFiles__}"

// {AppName} = CppScanner




size_t IO_GetFileSize(const char* Name)
{
	ASSERT(Name);
	int handle;

	handle = open(Name, O_RDONLY);
	if (handle == -1)
		return 0;

#ifdef _WIN32
	size_t size = _filelengthi64(handle);
#else
	struct stat st;
	fstat(handle, &st);
	off_t size = st.st_size;
#endif
	close(handle);

	return size;
}

/*
// @param must not be 0
size_t IO_GetFileSize(const wchar_t* filePath)
{
	ASSERT(filePath);

#ifdef _WIN32
	HANDLE handle = CreateFile(filePath,
		GENERIC_READ,
		0, // exclusive access
		NULL,
		OPEN_EXISTING,
		0,
		NULL);

	if (handle == INVALID_HANDLE_VALUE)
		return 0;

	DWORD lowFileSize = 0;
	DWORD highFileSize = 0;

	lowFileSize = GetFileSize(handle, &highFileSize);

	CloseHandle(handle);

	return (((size_t)highFileSize)<<32) | (size_t)lowFileSize;
#else
	ASSERT(0); // todo
	return 0;
#endif
}
*/


CASCIIFile::CASCIIFile()
{
	m_Size = 0;
	m_Data = 0;
}

// destructor
CASCIIFile::~CASCIIFile()
{
	ReleaseData();
}


// copy constructor
CASCIIFile::CASCIIFile(CASCIIFile& a)
{
	char* mem = (char*)malloc(a.m_Size);

	if (mem)
	{
		m_Data = mem;
		m_Size = a.m_Size;
	}
	else throw "low memory";
}


// assignment operator
const CASCIIFile CASCIIFile::operator=(const CASCIIFile& a)
{
	char* mem = (char*)malloc(a.m_Size);

	if (mem)
	{
		ReleaseData();
		m_Data = mem;
		m_Size = a.m_Size;
	}
	else throw "low memory";

	return *this;
}





void CASCIIFile::ReleaseData()
{
	free((void*)m_Data);
	m_Data = 0;
	m_Size = 0;
}

bool CASCIIFile::IO_GetAvailability(const char* pathname)
{
	int handle;

	handle = open(pathname, O_RDONLY);

	if (handle == -1)
		return false;

#ifdef _WIN32
	close(handle);
#else
	ASSERT(0); // todo
#endif

	return true;
}


/*
bool CASCIIFile::IO_SaveASCIIFile(const wchar_t* pathname)
{
	if (m_Data == 0)
		return false;

	FILE* file;

#ifdef _WIN32
	if ((file = _wfopen(pathname, L"wb")) != NULL)
	{
		fwrite(m_Data, m_Size - 1, 1, file);
		fclose(file);
		return true;
	}
#else
	ASSERT(0); // todo
#endif

	return false;
}
*/

bool CASCIIFile::IO_SaveASCIIFile(const char* pathname)
{
	if (m_Data == 0)
		return false;

	FILE* file;

	if ((file = fopen(pathname, "wb")) != NULL)
	{
		fwrite(m_Data, m_Size - 1, 1, file);
		fclose(file);
		return true;
	}

	return false;
}


bool CASCIIFile::IO_LoadASCIIFile(const char* pathname)
{
	if (*pathname == 0)
		return false;

	size_t size = IO_GetFileSize(pathname);
	char* ret = 0;

	ret = (char*)malloc(size + 1);

	if (ret == 0)
		return false;	// no memory

	FILE* file;

	if ((file = fopen(pathname, "rb")) != NULL)
	{
		fread(ret, size, 1, file);
		ret[size] = 0;				// 0 terminate

		m_Size = size + 1;
		m_Data = ret;

		fclose(file);
		return true;
	}

	free(ret);
	return false;
}

/*
bool CASCIIFile::IO_LoadASCIIFile(const wchar_t* pathname)
{
	if (*pathname == 0)
		return false;

	size_t size = IO_GetFileSize(pathname);
	char* ret = 0;

	ret = (char*)malloc(size + 1);

	if (ret == 0)
		return false;	// no memory

	FILE* file;

#ifdef _WIN32
	if ((file = _wfopen(pathname, L"rb")) != NULL)
	{
		fread(ret, size, 1, file);
		ret[size] = 0;				// 0 terminate

		m_Size = size + 1;
		m_Data = ret;

		fclose(file);
		return true;
	}
#else
	ASSERT(0); // todo
#endif

	free(ret);
	return false;
}
*/

size_t CASCIIFile::GetDataSize() const
{
	if (m_Data)return m_Size - 1;
	else return 0;
}

const char* CASCIIFile::GetDataPtr()
{
	return m_Data;
}

void CASCIIFile::CoverThisData(const char* ptr, const size_t size)
{
	ReleaseData();
	m_Data = ptr;
	m_Size = size + 1;
}

