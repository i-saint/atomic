#include "istPCH.h"
#include "DLL.h"
#include "ist/Base/FileLoader.h"

#ifdef ist_env_Windows
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "rpcrt4.lib")
#include <windows.h>
#include <psapi.h>
#include <rpc.h>

namespace ist {

istAPI stl::string GetMachineID()
{
    stl::string ret;
    char value[64];
    DWORD size = _countof(value);
    DWORD type = REG_SZ;
    HKEY key;
    LONG retKey = ::RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", 0, KEY_READ|KEY_WOW64_64KEY, &key);
    LONG retVal = ::RegQueryValueExA(key, "MachineGuid", nullptr, &type, (LPBYTE)value, &size );
    if( retKey==ERROR_SUCCESS && retVal==ERROR_SUCCESS  ) {
        ret = value;
    }
    ::RegCloseKey( key );
    return ret;
}

istAPI stl::string CreateUUID()
{
    UUID uuid;
    ::UuidCreate(&uuid);
    char value[64];
    istSPrintf(value, "%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X",
        uuid.Data1, uuid.Data2, uuid.Data3,
        uuid.Data4[0], uuid.Data4[1], uuid.Data4[2], uuid.Data4[3], uuid.Data4[4], uuid.Data4[5], uuid.Data4[6], uuid.Data4[7]);
    return value;
}


EnvironmentVariables::Value::Value(const char *name)
{
    m_name = name;
    m_value.resize(1024*32);
    DWORD ret = ::GetEnvironmentVariableA(name, &m_value[0], m_value.size());
    m_value.resize(ret);
}
EnvironmentVariables::Value::operator const char*() const
{
    return m_value.c_str();
}
void EnvironmentVariables::Value::operator=(const char *value)
{
    m_value = value;
    ::SetEnvironmentVariableA(m_name.c_str(), m_value.c_str());
}
void EnvironmentVariables::Value::operator+=(const char *value)
{
    m_value += value;
    ::SetEnvironmentVariableA(m_name.c_str(), m_value.c_str());
}

EnvironmentVariables::Value EnvironmentVariables::get(const char *name)
{
    return Value(name);
}


DLL::DLL()
    : m_mod(nullptr)
{
}
DLL::DLL(const char *path)
    : m_mod(nullptr)
{
    load(path);
}
DLL::~DLL()
{
    unload();
}
bool DLL::load(const char *path)
{
    unload();
    if(HMODULE m=::LoadLibraryA(path)) {
        m_mod = m;
        char tmp[MAX_PATH];
        ::GetModuleFileNameA(m, tmp, MAX_PATH);
        m_path = tmp;
        return true;
    }
    return false;
}
bool DLL::unload()
{
    if(m_mod) {
        m_mod = nullptr;
        m_path.clear();
        return true;
    }
    return false;
}
void* DLL::findSymbol(const char *name) const
{
    return ::GetProcAddress(m_mod, name);
}
void* DLL::getHandle() const
{
    return m_mod;
}
const std::string& DLL::getPath() const
{
    return m_path;
}




void EnumerateDependentModules( const char *path_to_dll_or_exe, const std::function<void (const char*)> &f )
{
    stl::string buf;
    if(!FileToString(path_to_dll_or_exe, buf)) { return; }

    size_t ImageBase = (size_t)&buf[0];
    PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)ImageBase;
    if(pDosHeader->e_magic!=IMAGE_DOS_SIGNATURE) { return; }

    PIMAGE_NT_HEADERS pNTHeader = (PIMAGE_NT_HEADERS)(ImageBase + pDosHeader->e_lfanew);
    DWORD RVAImports = pNTHeader->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
    if(RVAImports==0) { return; }

    PIMAGE_SECTION_HEADER pSectionHeader = IMAGE_FIRST_SECTION(pNTHeader);
    for(size_t i=0; i<pNTHeader->FileHeader.NumberOfSections; ++i) {
        PIMAGE_SECTION_HEADER s = pSectionHeader+i;
        if(RVAImports >= s->VirtualAddress && RVAImports < s->VirtualAddress+s->SizeOfRawData) {
            pSectionHeader = s;
            break;
        }
    }
    DWORD gap = pSectionHeader->VirtualAddress - pSectionHeader->PointerToRawData;

    IMAGE_IMPORT_DESCRIPTOR *pImportDesc = (IMAGE_IMPORT_DESCRIPTOR*)(ImageBase + RVAImports - gap);
    while(pImportDesc->Name!=0) {
        const char *pDLLName = (const char*)(ImageBase + pImportDesc->Name - gap);
        f(pDLLName);
        ++pImportDesc;
    }
}


template<class T>
static inline void ForceWrite(T &dst, const T &src)
{
    DWORD old_flag;
    ::VirtualProtect(&dst, sizeof(T), PAGE_EXECUTE_READWRITE, &old_flag);
    dst = src;
    ::VirtualProtect(&dst, sizeof(T), old_flag, &old_flag);
}

static inline bool IsValidMemory(void *p)
{
    if(p==NULL) { return false; }
    MEMORY_BASIC_INFORMATION meminfo;
    return ::VirtualQuery(p, &meminfo, sizeof(meminfo))!=0 && meminfo.State!=MEM_FREE;
}

void EnumerateDLLImports(HMODULE module, const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1,
    const std::function<void (DWORD ordinal, void *&func)> &f2 )
{
    if(!IsValidMemory(module)) { return; }

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

void EnumerateDLLImports(HMODULE module, const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1 )
{
    EnumerateDLLImports(module, dllfilter, f1, [](DWORD ordinal, void *&func){});
}


void EnumerateDLLImportsEveryModule(const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1,
    const std::function<void (DWORD ordinal, void *&func)> &f2 )
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
void EnumerateDLLImportsEveryModule(const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1 )
{
    EnumerateDLLImportsEveryModule(dllfilter, f1, [](DWORD ordinal, void *&func){});
}


// dll が export している関数 (への RVA) を書き換える
// それにより、GetProcAddress() が返す関数をすり替える
// 元の関数へのポインタを返す
void* OverrideDLLExport(HMODULE module, const char *funcname, void *replacement)
{
    if(!IsValidMemory(module)) { return NULL; }

    size_t ImageBase = (size_t)module;
    PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)ImageBase;
    if(pDosHeader->e_magic!=IMAGE_DOS_SIGNATURE) { return NULL; }

    PIMAGE_NT_HEADERS pNTHeader = (PIMAGE_NT_HEADERS)(ImageBase + pDosHeader->e_lfanew);
    DWORD RVAExports = pNTHeader->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
    if(RVAExports==0) { return NULL; }

    IMAGE_EXPORT_DIRECTORY *pExportDirectory = (IMAGE_EXPORT_DIRECTORY *)(ImageBase + RVAExports);
    DWORD *RVANames = (DWORD*)(ImageBase+pExportDirectory->AddressOfNames);
    WORD *RVANameOrdinals = (WORD*)(ImageBase+pExportDirectory->AddressOfNameOrdinals);
    DWORD *RVAFunctions = (DWORD*)(ImageBase+pExportDirectory->AddressOfFunctions);
    for(DWORD i=0; i<pExportDirectory->NumberOfFunctions; ++i) {
        char *pName = (char*)(ImageBase+RVANames[i]);
        if(strcmp(pName, funcname)==0) {
            void *before = (void*)(ImageBase+RVAFunctions[RVANameOrdinals[i]]);
            ForceWrite<DWORD>(RVAFunctions[RVANameOrdinals[i]], (DWORD)replacement - ImageBase);
            return before;
        }
    }
    return NULL;
}

} // namespace ist
#endif // ist_env_Windows
