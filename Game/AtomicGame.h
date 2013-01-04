#ifndef atomic_Game_AtomicGame_h
#define atomic_Game_AtomicGame_h
#include "Input.h"
#include "AtomicApplication.h"


namespace atomic {

class World;
class AtomicRenderer;

class AtomicGame
{
private:
    IInputServer    *m_input_server;
    World           *m_world;
    SFMT            m_rand;
#ifdef atomic_enable_sync_lock
    bool m_sync_lock;
#endif // atomic_enable_sync_lock

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

    int handleCommandLine(const stl::wstring &command);

    const InputState* getIngameInputs() const { return m_input_server->getInput(); }
    World* getWorld()   { return m_world; }
    SFMT* getRandom();

#ifdef atomic_enable_sync_lock
    void lockSyncMethods()              { m_sync_lock=true; }
    void unlockSyncMethods()            { m_sync_lock=false; }
    bool isSyncMethodsEnabled() const   { return m_sync_lock; }
#endif // atomic_enable_sync_lock
};

#define atomicGetWorld()            atomicGetGame()->getWorld()
#define atomicGetIngameInputs()     atomicGetGame()->getIngameInputs()
#define atomicGetRandom()           atomicGetGame()->getRandom()

#ifdef atomic_enable_sync_lock
#   define  atomicLockSyncMethods()     atomicGetGame()->lockSyncMethods()
#   define  atomicUnlockSyncMethods()   atomicGetGame()->unlockSyncMethods()
#   define  atomicAssertSyncLock(...)   istAssert(!atomicGetGame()->isSyncMethodsEnabled(), __VA_ARGS__)
#else
#   define  atomicLockSyncMethods()     
#   define  atomicUnlockSyncMethods()   
#   define  atomicAssertSyncLock(...)   
#endif // atomic_enable_sync_lock

} // namespace atomic
#endif atomic_Game_AtomicGame_h
