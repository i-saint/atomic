#include "launcherPCH.h"

istImplementOperatorNewDelete();

namespace atm {


void glob( const char *path, const char *filter_regex, const std::function<void (const std::string &file)> &f)
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
        std::string module_dir;
        std::string current_dir;
        {
            char buf[MAX_PATH + 1];
            HMODULE mod = 0;
            ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCSTR)&glob, &mod);
            DWORD size = ::GetModuleFileNameA(mod, buf, sizeof(buf));
            for (int i = size - 1; i >= 0; --i) {
                if (buf[i] == '\\') {
                    buf[i] = '\0';
                    break;
                }
            }
            module_dir = buf;
        }
        {
            char buf[MAX_PATH + 1];
            GetCurrentDirectoryA(sizeof(buf), buf);
            current_dir = buf;
        }

        // 環境変数 PATH に Binaries/Common を追加し、dll サーチパスに加える
        {

            std::string path;
            //path.resize(1024*64);
            //DWORD ret = ::GetEnvironmentVariableA("PATH", &path[0], path.size());
            path += module_dir;
            path += "\\Binaries\\Common;";
            path += current_dir;
            path += "\\Binaries\\Common;";
            ::SetEnvironmentVariableA("PATH", path.c_str());
        }

#ifdef ist_env_Master
        const char *engine = "Binaries\\atomic_engine.dll";
#elif defined(ist_env_Debug)
        const char *engine = "Binaries\\atomic_engine_dbg.dll";
#else
        const char *engine = "Binaries\\atomic_engine_dev.dll";
#endif
        m_enginedll = ::LoadLibraryA(engine);
        if (m_enginedll == nullptr) {
            m_enginedll = ::LoadLibraryA((module_dir + "\\" + engine).c_str());
        }
        if (m_enginedll == nullptr) {
            m_enginedll = ::LoadLibraryA((current_dir + "\\" + engine).c_str());
        }
    }

    ~Launcher()
    {
        each(m_commondlls, [&](HMODULE dll){ ::FreeLibrary(dll); });
    }

    __declspec(noinline) int32 run(int argc, char* argv[])
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
