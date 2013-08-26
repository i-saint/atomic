#ifndef atm_Engine_Game_PluinManager_h
#define atm_Engine_Game_PluinManager_h

namespace atm {

#ifdef atm_enable_Plugin
class PluginManager
{
public:
    static bool initializeInstance();
    static bool finalizeInstance();
    static PluginManager* getInstance();

private:
    PluginManager();
    ~PluginManager();

    static PluginManager *s_inst;
    ist::vector<HMODULE> m_dlls;
};

#define atmPluginManagerInitialize()    PluginManager::initializeInstance()
#define atmPluginManagerFinalize()      PluginManager::finalizeInstance()
#else  // atm_enable_Plugin

#define atmPluginManagerInitialize()    
#define atmPluginManagerFinalize()      

#endif // atm_enable_Plugin

} // namespace atm
#endif // atm_Engine_Game_PluinManager_h
