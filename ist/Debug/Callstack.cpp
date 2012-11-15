#include "istPCH.h"
#include "ist/Base/CRTex.h"
#include "Callstack.h"

#ifdef ist_env_Windows
#   include <windows.h>
#   include <dbghelp.h>
#   pragma comment(lib, "dbghelp.lib")
#endif // ist_env_Windows

namespace ist {

bool InitializeDebugSymbol()
{
#ifdef ist_env_Windows
    if(!::SymInitialize(::GetCurrentProcess(), NULL, TRUE)) {
        return false;
    }
    ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);

    return true;
#else // ist_env_Windows
    return false;
#endif // ist_env_Windows
}

void FinalizeDebugSymbol()
{
#ifdef ist_env_Windows
    ::SymCleanup(::GetCurrentProcess());
#else // ist_env_Windows
#endif // ist_env_Windows
}


int GetCallstack(void **callstack, int callstack_size, int skip_size)
{
#ifdef ist_env_Windows
    return CaptureStackBackTrace(skip_size, callstack_size, callstack, NULL);
#else // ist_env_Windows
#endif // ist_env_Windows
}

stl::string AddressToSymbolName(void *address)
{
#ifdef ist_env_Windows

#ifdef _WIN64
    typedef DWORD64 DWORDX;
    typedef PDWORD64 PDWORDX;
#else
    typedef DWORD DWORDX;
    typedef PDWORD PDWORDX;
#endif

    char buf[1024];
    HANDLE process = ::GetCurrentProcess();
    IMAGEHLP_MODULE imageModule = { sizeof(IMAGEHLP_MODULE) };
    IMAGEHLP_LINE line ={sizeof(IMAGEHLP_LINE)};
    DWORDX dispSym = 0;
    DWORD dispLine = 0;

    char symbolBuffer[sizeof(IMAGEHLP_SYMBOL) + MAX_PATH] = {0};
    IMAGEHLP_SYMBOL * imageSymbol = (IMAGEHLP_SYMBOL*)symbolBuffer;
    imageSymbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
    imageSymbol->MaxNameLength = MAX_PATH;

    if(!::SymGetModuleInfo(process, (DWORDX)address, &imageModule)) {
        istsprintf(buf, "[0x%p]\n", address);
    }
    else if(!::SymGetSymFromAddr(process, (DWORDX)address, &dispSym, imageSymbol)) {
        istsprintf(buf, "%s + 0x%x [0x%p]\n", imageModule.ModuleName, ((size_t)address-(size_t)imageModule.BaseOfImage), address);
    }
    else if(!::SymGetLineFromAddr(process, (DWORDX)address, &dispLine, &line)) {
        istsprintf(buf, "%s!%s + 0x%x [0x%p]\n", imageModule.ModuleName, imageSymbol->Name, ((size_t)address-(size_t)imageSymbol->Address), address);
    }
    else {
        istsprintf(buf, "%s(%d): %s!%s + 0x%x [0x%p]\n", line.FileName, line.LineNumber,
            imageModule.ModuleName, imageSymbol->Name, ((size_t)address-(size_t)imageSymbol->Address), address);
    }
    return buf;

#else // ist_env_Windows
    return "";
#endif // ist_env_Windows
}

stl::string CallstackToSymbolNames(void **callstack, int callstack_size, int clamp_head, int clamp_tail, const char *indent)
{
#ifdef ist_env_Windows
    stl::string tmp;
    int begin = stl::max<int>(0, clamp_head);
    int end = stl::max<int>(0, callstack_size-clamp_tail);
    for(int i=begin; i<end; ++i) {
        tmp += indent;
        tmp += AddressToSymbolName(callstack[i]);
    }
    return tmp;
#else // ist_env_Windows

    return "";
#endif // ist_env_Windows
}

} // namespace ist
