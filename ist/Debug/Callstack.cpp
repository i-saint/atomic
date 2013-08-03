#include "istPCH.h"
#include "ist/stdex/crtex.h"
#include "Callstack.h"

#ifdef ist_env_Windows
#   include <windows.h>
#   include <dbghelp.h>
#   include <psapi.h>
#   pragma comment(lib, "dbghelp.lib")
#   pragma comment(lib, "psapi.lib")
#endif // ist_env_Windows

namespace ist {

#ifdef ist_env_Windows

istAPI bool InitializeDebugSymbol()
{
    if(!::SymInitialize(::GetCurrentProcess(), NULL, TRUE)) {
        return false;
    }
    ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    return true;
}

istAPI void FinalizeDebugSymbol()
{
    ::SymCleanup(::GetCurrentProcess());
}

istAPI int GetCallstack(void **callstack, int callstack_size, int skip_size)
{
    return CaptureStackBackTrace(skip_size, callstack_size, callstack, NULL);
}

istAPI stl::string AddressToSymbolName(void *address)
{
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
        istSPrintf(buf, "[0x%p]\n", address);
    }
    else if(!::SymGetSymFromAddr(process, (DWORDX)address, &dispSym, imageSymbol)) {
        istSPrintf(buf, "%s + 0x%x [0x%p]\n", imageModule.ModuleName, ((size_t)address-(size_t)imageModule.BaseOfImage), address);
    }
    else if(!::SymGetLineFromAddr(process, (DWORDX)address, &dispLine, &line)) {
        istSPrintf(buf, "%s!%s + 0x%x [0x%p]\n", imageModule.ModuleName, imageSymbol->Name, ((size_t)address-(size_t)imageSymbol->Address), address);
    }
    else {
        istSPrintf(buf, "%s(%d): %s!%s + 0x%x [0x%p]\n", line.FileName, line.LineNumber,
            imageModule.ModuleName, imageSymbol->Name, ((size_t)address-(size_t)imageSymbol->Address), address);
    }
    return buf;
}

istAPI stl::string CallstackToSymbolNames(void **callstack, int callstack_size, int clamp_head, int clamp_tail, const char *indent)
{
    stl::string tmp;
    int begin = stl::max<int>(0, clamp_head);
    int end = stl::max<int>(0, callstack_size-clamp_tail);
    for(int i=begin; i<end; ++i) {
        tmp += indent;
        tmp += AddressToSymbolName(callstack[i]);
    }
    return tmp;
}


istAPI bool IsStaticMemory(void *addr)
{
    if(addr==NULL) { return false; }
    // static 領域はモジュール (exe,dll) が map されている領域内にある
    // 高速化のため呼び出し元モジュールのみ調べる
    // 他モジュールも調べる場合 ::EnumProcessModules() とかを使う
    MODULEINFO modinfo;
    {
        HMODULE mod = 0;
        void *retaddr = *(void**)_AddressOfReturnAddress();
        ::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCTSTR)retaddr, &mod);
        ::GetModuleInformation(::GetCurrentProcess(), mod, &modinfo, sizeof(modinfo));
    }
    return addr>=modinfo.lpBaseOfDll && addr<reinterpret_cast<char*>(modinfo.lpBaseOfDll)+modinfo.SizeOfImage;
}

istAPI bool IsStackMemory(void *addr)
{
    if(addr==NULL) { return false; }
    // Thread Information Block に上限下限情報が入っている
    // (これだと現在のスレッドの stack 領域しか判別できない。
    //  別スレッドの stack かも調べたい場合のいい方法がよくわからず。
    //  ::Thread32First(), ::Thread32Next() で全プロセスの全スレッドを巡回するしかない…？)
    NT_TIB *tib = reinterpret_cast<NT_TIB*>(::NtCurrentTeb());
    return addr>=tib->StackLimit && addr<tib->StackBase;
}

istAPI bool IsHeapMemory(void *addr)
{
    if(addr==NULL) { return false; }
    // static 領域ではない && stack 領域でもない && 有効なメモリ (::VirtualQuery() が成功する) なら true
    // ::HeapWalk() で照合するのが礼儀正しいアプローチだが、
    // こっちの方が速いし、別スレッドや別モジュールから呼び出されるのでなければ結果も正しいはず
    MEMORY_BASIC_INFORMATION meminfo;
    return !IsStackMemory(addr) && !IsStaticMemory(addr) && 
        ::VirtualQuery(addr, &meminfo, sizeof(meminfo))>0 && meminfo.State!=MEM_FREE;
}


static BOOL CALLBACK _CB_GetThisOfCaller( SYMBOL_INFO* si, ULONG size, PVOID p )
{
    if(si && si->NameLen==4 && strncmp(si->Name, "this", 4)==0) {
        auto *ret = (std::pair<ULONG64,bool>*)p;
        ret->first = si->Address;
        ret->second = true;
        return FALSE;
    }
    return TRUE;
}

istAPI void* GetThisOfCaller()
{
    // thanks to http://jpassing.com/2008/03/12/walking-the-stack-of-the-current-thread/
    CONTEXT context;
#ifdef _WIN64
    ::RtlCaptureContext(&context);
#else
    ::ZeroMemory( &context, sizeof(context) );
    context.ContextFlags = CONTEXT_CONTROL;
    __asm
    {
        EIP:
        mov [context.Ebp], ebp;
        mov [context.Esp], esp;
        mov eax, [EIP];
        mov [context.Eip], eax;
    }
#endif 

    STACKFRAME64 stackFrame;
    ::ZeroMemory( &stackFrame, sizeof(stackFrame) );
#ifdef _WIN64
    DWORD machineType = IMAGE_FILE_MACHINE_AMD64;
    stackFrame.AddrPC.Offset = context.Rip;
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Offset = context.Rbp;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Offset = context.Rsp;
    stackFrame.AddrStack.Mode = AddrModeFlat;
#else
    DWORD machineType = IMAGE_FILE_MACHINE_I386;
    stackFrame.AddrPC.Offset = context.Eip;
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Offset = context.Ebp;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Offset = context.Esp;
    stackFrame.AddrStack.Mode = AddrModeFlat;
#endif

    HANDLE hProcess = ::GetCurrentProcess();
    HANDLE hThread = ::GetCurrentThread();
    ::StackWalk64(machineType, hProcess, hThread, &stackFrame, &context, NULL, NULL, NULL, NULL); // この関数のスタックフレーム
    ::StackWalk64(machineType, hProcess, hThread, &stackFrame, &context, NULL, NULL, NULL, NULL); // 呼び出し元
    ::StackWalk64(machineType, hProcess, hThread, &stackFrame, &context, NULL, NULL, NULL, NULL); // 呼び出し元の呼び出し元 (ターゲット)

    std::pair<ULONG64,bool> ret(0,false);
    IMAGEHLP_STACK_FRAME sf; 
    sf.ReturnOffset = stackFrame.AddrReturn.Offset;
    sf.FrameOffset = stackFrame.AddrFrame.Offset;
    sf.StackOffset = stackFrame.AddrStack.Offset;
    sf.InstructionOffset = stackFrame.AddrPC.Offset;
    ::SymSetContext(hProcess, &sf, 0 );
    ::SymEnumSymbols(hProcess, 0, 0, _CB_GetThisOfCaller, &ret);

    if(!ret.second) { return NULL; }
#ifdef _WIN64
    return *(void**)(stackFrame.AddrStack.Offset + ret.first);
#else
    return *(void**)(stackFrame.AddrFrame.Offset + ret.first);
#endif
}
#endif // ist_env_Windows

} // namespace ist
