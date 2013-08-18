#ifndef ist_System_DLL_h
#define ist_System_DLL_h
namespace ist {

#ifdef ist_env_Windows

void EnumerateDependentModules(const char *path_to_dll_or_exe, const std::function<void (const char*)> &f);

void EnumerateDLLImports(HMODULE module, const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1,
    const std::function<void (DWORD ordinal, void *&func)> &f2 );

void EnumerateDLLImports(HMODULE module, const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1 );


void EnumerateDLLImportsEveryModule(const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1,
    const std::function<void (DWORD ordinal, void *&func)> &f2 );

void EnumerateDLLImportsEveryModule(const char *dllfilter,
    const std::function<void (const char*, void *&)> &f1 );


void* OverrideDLLExport(HMODULE module, const char *funcname, void *replacement);

#endif // ist_env_Windows

} // namespace ist
#endif // 
