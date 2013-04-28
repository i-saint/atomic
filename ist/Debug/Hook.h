#ifndef ist_Debug_Hook_h
#define ist_Debug_Hook_h
#ifdef ist_env_Windows

namespace ist {

// target: 関数ポインタ。対象関数を hotpatch して元の関数へのポインタを返す
void* Hotpatch( void *target, const void *replacement );
void* UglyHotpatch( void *target, const void *replacement );

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

} // namespace ist
#endif // ist_env_Windows
#endif // ist_Debug_Hook_h
