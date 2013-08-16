#include "stdafx.h"

istImplementOperatorNewDelete();

namespace atm {


atmAPI void glob( const char *path, const char *filter_regex, const std::function<void (const std::string &file)> &f)
{
    try {
        std::regex filter(filter_regex);
        Poco::DirectoryIterator end;
        for(Poco::DirectoryIterator i(Poco::Path((const char*)path)); i!=end; ++i) {
            if(std::regex_search(i->path(), filter)) {
                f(i->path());
            }
        }
    }
    catch(...) {
    }
}

template<class T, class F>
inline void each(T &v, const F &f) {
    std::for_each(v.begin(), v.end(), f);
}


class Launcher
{
private:
    ist::vector<HMODULE> m_commondlls;
    HMODULE m_enginedll;

public:
    Launcher() : m_commondlls(), m_enginedll()
    {
        //// todo: 依存関係がある dll もスマートに解決したい。SetDllDirectoryA() では無理？
        //::SetDllDirectoryA("Binaries");
        for(int i=0; i<2; ++i) {
            glob("Binaries\\Common", "\\.dll$", [&](const std::string &file){
                if(HMODULE dll = ::LoadLibraryA(file.c_str())) {
                    m_commondlls.push_back(dll);
                }
            });
        }

#ifdef ist_env_Master
        const char *engine = "Binaries\\atomic_engine.dll";
#elif defined(ist_env_Debug)
        const char *engine = "Binaries\\atomic_engine_dbg.dll";
#else
        const char *engine = "Binaries\\atomic_engine_dev.dll";
#endif
        m_enginedll = ::LoadLibraryA(engine);
    }

    ~Launcher()
    {
        each(m_commondlls, [&](HMODULE dll){ ::FreeLibrary(dll); });
    }

    int32 run(int argc, char* argv[])
    {
        typedef int (*EntryPoint)(int argc, char *argv[]);
        if(EntryPoint e = (EntryPoint)::GetProcAddress(m_enginedll, "atmMain")) {
            return e(argc, argv);
        }
        return -1;
    }
};
} // namespace atm
using namespace atm;

int istmain(int argc, char* argv[])
{
    Launcher *launcher = istNew(Launcher)();
    int32 ret = launcher->run(argc, argv);
    istDelete(launcher);
    return ret;
}
