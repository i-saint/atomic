#include "istPCH.h"
#include "ist/Base/CRTex.h"
#include "CrashDump.h"

#if defined(ist_enable_CrashReport) && defined(ist_env_Windows)
#pragma comment ( lib, "dbghelp.lib" )

namespace ist {

static DumpFileHandler g_dump_file_handler;

void SetDumpFileHandler( const DumpFileHandler &dh )
{
    g_dump_file_handler = dh;
}

void HandleCrash(EXCEPTION_POINTERS* ep)
{
    char filename[128];
    istsnprintf(filename, _countof(filename), "crashreport_%u.dmp", (uint32)::time(0));
    HANDLE hFile = ::CreateFileA(filename, GENERIC_READ | GENERIC_WRITE, 
        0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL ); 

    if((hFile!=NULL) && (hFile!=INVALID_HANDLE_VALUE))  {
        MINIDUMP_EXCEPTION_INFORMATION mdei; 
        mdei.ThreadId           = ::GetCurrentThreadId(); 
        mdei.ExceptionPointers  = ep; 
        mdei.ClientPointers     = FALSE; 

        MINIDUMP_CALLBACK_INFORMATION mci; 
        mci.CallbackRoutine     = (MINIDUMP_CALLBACK_ROUTINE)NULL; 
        mci.CallbackParam       = 0; 

        MINIDUMP_TYPE mdt       = (MINIDUMP_TYPE)(MiniDumpWithIndirectlyReferencedMemory | MiniDumpScanMemory); 

        BOOL rv = ::MiniDumpWriteDump( ::GetCurrentProcess(), ::GetCurrentProcessId(), 
            hFile, mdt, (ep!=0) ? &mdei : 0, 0, &mci ); 
        if(rv && g_dump_file_handler) {
            g_dump_file_handler(filename);
        }

        ::CloseHandle( hFile ); 
    }
}

} // namespace ist
#endif // defined(ist_enable_CrashReport) && defined(ist_env_Windows)
