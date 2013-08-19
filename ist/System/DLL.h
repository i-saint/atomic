#ifndef ist_System_DLL_h
#define ist_System_DLL_h
namespace ist {

#ifdef ist_env_Windows


class EnvironmentVariables
{
public:
    class Value
    {
    public:
        Value(const char *name);
        operator const char*() const;
        void operator=(const char *value);
        void operator+=(const char *value);
    private:
        stl::string m_name;
        stl::string m_value;
    };
public:
    static Value get(const char *name);
};

class DLL
{
public:
    DLL();
    DLL(const char *path);
    ~DLL();
    bool load(const char *path);
    bool unload();
    void* findSymbol(const char *path) const;
    void* getHandle() const;
    const std::string& getPath() const;
private:
    HMODULE m_mod;
    std::string m_path;
};


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
