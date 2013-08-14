#include "stdafx.h"
#include "PluginManager.h"
#include "Util.h"

namespace atm {

#ifdef atm_enable_Plugin
PluginManager* PluginManager::s_inst = nullptr;

bool PluginManager::initializeInstance()
{
    if(!s_inst) {
        s_inst = istNew(PluginManager)();
        return true;
    }
    return false;
}

bool PluginManager::finalizeInstance()
{
    if(s_inst) {
        istSafeDelete(s_inst);
        return true;
    }
    return false;
}

PluginManager* PluginManager::getInstance()
{
    return s_inst;
}


PluginManager::PluginManager()
{
    int32 num_loaded = 0;
    glob("Resources", "\\.dll$", [&](const std::string &file){
        if(HMODULE dll = ::LoadLibraryA(file.c_str())) {
            m_dlls.push_back(dll);
#ifndef dpDisable
            ++num_loaded;
            DWORD64 base = 0x10000000 + 0x1000000*num_loaded;
            if(!::SymLoadModuleEx(::GetCurrentProcess(), nullptr, file.c_str(), nullptr, base, 0, nullptr, 0)) {
                istPrint("SymLoadModuleEx() for %s failed\n", file.c_str());
            }
#endif // dpDisable
        }
    });
}

PluginManager::~PluginManager()
{
    each(m_dlls, [&](HMODULE h){
        ::FreeLibrary(h);
    });
}
#endif // atm_enable_Plugin

} // namespace atm
