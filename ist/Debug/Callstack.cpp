#include "istPCH.h"
#include "ist/Base/CRTex.h"
#include "Callstack.h"

#ifdef __ist_env_Windows__
#   include <windows.h>
#   include <imagehlp.h>
#   pragma comment(lib, "imagehlp.lib")
#endif // __ist_env_Windows__

namespace ist {

bool InitializeDebugSymbol()
{
#ifdef __ist_env_Windows__
    if(!::SymInitialize(::GetCurrentProcess(), NULL, TRUE)) {
        return false;
    }
    ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);

    return true;
#else // __ist_env_Windows__
    return false;
#endif // __ist_env_Windows__
}

void FinalizeDebugSymbol()
{
#ifdef __ist_env_Windows__
    ::SymCleanup(::GetCurrentProcess());
#else // __ist_env_Windows__
#endif // __ist_env_Windows__
}


int GetCallstack(void **callstack, int callstack_size, int skip_size)
{
#ifdef __ist_env_Windows__
    return CaptureStackBackTrace(skip_size, callstack_size, callstack, NULL);
#else // __ist_env_Windows__
#endif // __ist_env_Windows__
}

stl::string AddressToSymbolName(void *address)
{
#ifdef __ist_env_Windows__

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

#else // __ist_env_Windows__
    return "";
#endif // __ist_env_Windows__
}

stl::string CallstackToSymbolNames(void **callstack, int callstack_size, int clamp_head, int clamp_tail, const char *indent)
{
#ifdef __ist_env_Windows__
    stl::string tmp;
    int begin = stl::max<int>(0, clamp_head);
    int end = stl::max<int>(0, callstack_size-clamp_tail);
    for(int i=begin; i<end; ++i) {
        tmp += indent;
        tmp += AddressToSymbolName(callstack[i]);
    }
    return tmp;
#else // __ist_env_Windows__

    return "";
#endif // __ist_env_Windows__
}

} // namespace ist
