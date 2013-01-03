#ifndef __atomic_Game_AtomicGame__
#define __atomic_Game_AtomicGame__
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
#ifdef atomic_enable_debug_rand_lock
    bool m_rand_lock;
#endif // __atomic_enable_debug_rand_lock__

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
    SFMT* getRandom()
    {
#ifdef atomic_enable_debug_rand_lock
        istAssert(m_rand_lock==false, "getRandom() is called from asycupdate.\n");
#endif // __atomic_enable_debug_rand_lock__
        return &m_rand;
    }

#ifdef atomic_enable_debug_rand_lock
    void lockRandom()   { m_rand_lock=true; }
    void unlockRandom() { m_rand_lock=false; }
#endif // __atomic_enable_debug_rand_lock__
};

#define atomicGetWorld()            atomicGetGame()->getWorld()
#define atomicGetIngameInputs()     atomicGetGame()->getIngameInputs()
#define atomicGetRandom()           atomicGetGame()->getRandom()

#ifdef atomic_enable_debug_rand_lock
#   define  atomicLockRandom()      atomicGetGame()->lockRandom()
#   define  atomicUnlockRandom()    atomicGetGame()->unlockRandom()
#else
#   define  atomicLockRandom()      
#   define  atomicUnlockRandom()    
#endif // __atomic_enable_debug_rand_lock__

} // namespace atomic
#endif __atomic_Game_AtomicGame__
