#include "stdafx.h"
#include "ist/Base/New.h"
#include "ist/Base/Assert.h"
#include "ist/Concurrency/ThreadUtil.h"

#ifdef _WIN32
#include <windows.h>

const DWORD MS_VC_EXCEPTION=0x406D1388;

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType; // Must be 0x1000.
    LPCSTR szName; // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

#endif // _WIN32


namespace ist {

void SetThreadName( uint32 dwThreadID, const char* name)
{
#ifdef _WIN32
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = name;
    info.dwThreadID = dwThreadID;
    info.dwFlags = 0;

    __try
    {
        RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
    }
#endif // _WIN32
}

void SetThreadName(const char *name)
{
#ifdef _WIN32
    SetThreadName((uint32_t)::GetCurrentThreadId(), name);
#endif // _WIN32
}

} // namespace ist
