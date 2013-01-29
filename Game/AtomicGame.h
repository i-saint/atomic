#ifndef atomic_Game_AtomicGame_h
#define atomic_Game_AtomicGame_h
#include "Input.h"
#include "AtomicApplication.h"
#include "Game/DebugMenu.h"
#include "Network/LevelEditorServer.h"


namespace atomic {

class World;
class AtomicRenderer;

class AtomicGame
{
public:
    AtomicGame();
    ~AtomicGame();

    bool readReplayFromFile(const char *path);

    void frameBegin();
    void update(float32 dt);
    void asyncupdateBegin(float32 dt);
    void asyncupdateEnd();
    void draw();
    void frameEnd();

    // 描画スレッドから呼ばれる
    void drawCallback();

    void handleLevelEditorCommand(const LevelEditorCommand &cmd);
    int handleCommandLine(const stl::wstring &command);

    const InputState* getIngameInputs() const { return m_input_server->getInput(); }
    World* getWorld()   { return m_world; }
    SFMT* getRandom();

#ifdef atomic_enable_sync_lock
    void dbgLockSyncMethods()               { m_sync_lock=true; }
    void dbgUnlockSyncMethods()             { m_sync_lock=false; }
    bool dbgIsSyncMethodsEnabled() const    { return m_sync_lock; }
#endif // atomic_enable_sync_lock

private:
    IInputServer    *m_input_server;
    World           *m_world;
    SFMT            m_rand;
#ifdef atomic_enable_sync_lock
    bool m_sync_lock;
#endif // atomic_enable_sync_lock
};

#define atomicGetWorld()            atomicGetGame()->getWorld()
#define atomicGetIngameInputs()     atomicGetGame()->getIngameInputs()
#define atomicGetRandom()           atomicGetGame()->getRandom()

#ifdef atomic_enable_sync_lock
#   define  atomicDbgLockSyncMethods()      atomicGetGame()->dbgLockSyncMethods()
#   define  atomicDbgUnlockSyncMethods()    atomicGetGame()->dbgUnlockSyncMethods()
#   define  atomicDbgAssertSyncLock(...)    istAssert(!atomicGetGame()->dbgIsSyncMethodsEnabled(), __VA_ARGS__)
#else
#   define  atomicDbgLockSyncMethods()      
#   define  atomicDbgUnlockSyncMethods()    
#   define  atomicDbgAssertSyncLock(...)    
#endif // atomic_enable_sync_lock

} // namespace atomic
#endif atomic_Game_AtomicGame_h
