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

int HandleCrash(EXCEPTION_POINTERS* ep)
{
    //if(!ep || ::IsDebuggerPresent()) {
    //    return EXCEPTION_CONTINUE_SEARCH;
    //}

    char filename[128];
    istSNPrintf(filename, _countof(filename), "crashdump_%u.dmp", (uint32)::time(0));
    HANDLE hFile = ::CreateFileA(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );

    if(hFile!=NULL && hFile!=INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mdei;
        mdei.ThreadId           = ::GetCurrentThreadId();
        mdei.ExceptionPointers  = ep;
        mdei.ClientPointers     = FALSE;

        MINIDUMP_CALLBACK_INFORMATION mci;
        mci.CallbackRoutine     = (MINIDUMP_CALLBACK_ROUTINE)NULL;
        mci.CallbackParam       = 0;

        MINIDUMP_TYPE mdt       = (MINIDUMP_TYPE)(MiniDumpWithIndirectlyReferencedMemory | MiniDumpScanMemory);

        BOOL rv = ::MiniDumpWriteDump( ::GetCurrentProcess(), ::GetCurrentProcessId(), hFile, mdt, &mdei, 0, &mci );
        ::CloseHandle( hFile );

        if(rv && g_dump_file_handler) {
            g_dump_file_handler(filename);
        }
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

} // namespace ist
#endif // defined(ist_enable_CrashReport) && defined(ist_env_Windows)
