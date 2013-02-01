#ifndef ist_Debug_CrashDump_h
#define ist_Debug_CrashDump_h

#include "ist/Config.h"
#ifdef ist_enable_CrashReport
#ifdef ist_env_Windows
#   include "dbghelp.h"
#endif // ist_env_Windows
#endif // ist_enable_CrashReport


#if defined(ist_enable_CrashReport) && defined(ist_env_Windows)

namespace ist {
    typedef std::function<void (const char *path)> DumpFileHandler;
    void SetDumpFileHandler(const DumpFileHandler &dh);
    int HandleCrash(EXCEPTION_POINTERS* ep);
} // namespace ist

#define istSetDumpFileHanlder(v)    ist::SetDumpFileHandler(v)
#define istCrashReportBegin         __try {
#define istCrashReportRescue        } __except( ist::HandleCrash(GetExceptionInformation()) ) {
#define istCrashReportEnd           }

#else // defined(ist_enable_CrashReport) && defined(ist_env_Windows)

#define istSetDumpFileHanlder(v)    
#define istCrashReportBegin         
#define istCrashReportRescue        
#define istCrashReportEnd           

#endif // defined(ist_enable_CrashReport) && defined(ist_env_Windows)

#endif // ist_Debug_CrashDump_h
