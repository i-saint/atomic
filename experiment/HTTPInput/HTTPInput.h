#ifndef HTTPInput_h
#define HTTPInput_h
#pragma comment(lib, "psapi.lib")
#include <psapi.h>
#include <stdint.h>
#include <vector>
#include <regex>

typedef char            int8;
typedef short           int16;
typedef int             int32;
typedef unsigned char   uint8;
typedef unsigned short  uint16;
typedef unsigned int    uint32;

extern "C" {
    struct HTTPInputData
    {
        struct Keyboard
        {
            char keys[256];
        };

        struct Mouse
        {
            int32 x,y;
            uint32 buttons;
        };

        struct Pad
        {
            int32 x1,y1;
            int32 x2,y2;
            int32 pov;
            uint32 buttons;
        };

        Keyboard key;
        Mouse    mouse;
        Pad      pad;
    };

    __declspec(dllexport) bool StartHTTPInputServer();
    __declspec(dllexport) bool StopHTTPInputServer();
    __declspec(dllexport) HTTPInputData* GetHTTPInputData();
} // extern "C"


// F1: [](const char *funcname, void *&func) {...}
// F2: [](DWORD ordinal, void *&func) {...}
template<class F1, class F2>
inline void EnumerateDLLImports(HMODULE module, const char *dllfilter, const F1 &f1, const F2 &f2)
{
    if(module==NULL) { return; }

    size_t ImageBase = (size_t)module;
    PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)ImageBase;
    if(pDosHeader->e_magic!=IMAGE_DOS_SIGNATURE) { return; }

    PIMAGE_NT_HEADERS pNTHeader = (PIMAGE_NT_HEADERS)(ImageBase + pDosHeader->e_lfanew);
    DWORD RVAImports = pNTHeader->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
    if(RVAImports==0) { return; }

    std::regex reg(dllfilter, std::regex::ECMAScript|std::regex::icase);
    IMAGE_IMPORT_DESCRIPTOR *pImportDesc = (IMAGE_IMPORT_DESCRIPTOR*)(ImageBase + RVAImports);
    while(pImportDesc->Name!=0) {
        const char *pDLLName = (const char*)(ImageBase+pImportDesc->Name);
        if(std::regex_match(pDLLName, reg)) {
            IMAGE_THUNK_DATA* pThunkOrig = (IMAGE_THUNK_DATA*)(ImageBase + pImportDesc->OriginalFirstThunk);
            IMAGE_THUNK_DATA* pThunk = (IMAGE_THUNK_DATA*)(ImageBase + pImportDesc->FirstThunk);
            while(pThunkOrig->u1.AddressOfData!=0) {
                if((pThunkOrig->u1.Ordinal & 0x80000000) > 0) {
                    DWORD Ordinal = pThunkOrig->u1.Ordinal & 0xffff;
                    f2(Ordinal, *(void**)pThunk);
                }
                else {
                    IMAGE_IMPORT_BY_NAME* pIBN = (IMAGE_IMPORT_BY_NAME*)(ImageBase + pThunkOrig->u1.AddressOfData);
                    f1((char*)pIBN->Name, *(void**)pThunk);
                }
                ++pThunkOrig;
                ++pThunk;
            }
        }
        ++pImportDesc;
    }
    return;
}
template<class F1>
inline void EnumerateDLLImports(HMODULE module, const char *dllfilter, const F1 &f1)
{
    EnumerateDLLImports(module, dllfilter, f1, [](DWORD ordinal, void *&func){});
}

template<class F1, class F2>
inline void EachImportFunctionInEveryModule(const char *dllfilter, const F1 &f1, const F2 &f2)
{
    std::vector<HMODULE> modules;
    DWORD num_modules;
    ::EnumProcessModules(::GetCurrentProcess(), NULL, 0, &num_modules);
    modules.resize(num_modules/sizeof(HMODULE));
    ::EnumProcessModules(::GetCurrentProcess(), &modules[0], num_modules, &num_modules);
    for(size_t i=0; i<modules.size(); ++i) {
        EnumerateDLLImports(modules[i], dllfilter, f1, f2);
    }
}
template<class F1>
inline void EachImportFunctionInEveryModule(const char *dllfilter, const F1 &f1)
{
    EachImportFunctionInEveryModule(dllfilter, f1, [](DWORD ordinal, void *&func){});
}


template<class T> inline void ForceWrite(T &dst, const T &src)
{
    DWORD old_flag;
    ::VirtualProtect(&dst, sizeof(T), PAGE_EXECUTE_READWRITE, &old_flag);
    dst = src;
    ::VirtualProtect(&dst, sizeof(T), old_flag, &old_flag);
}
#endif // HTTPInput_h
