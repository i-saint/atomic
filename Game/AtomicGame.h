#ifndef __atomic_Game_AtomicGame__
#define __atomic_Game_AtomicGame__
#include "Input.h"

namespace atomic {

class World;
class AtomicRenderer;

class AtomicGame
{
private:
    IInputServer    *m_input_server;
    World           *m_world;
    SFMT            m_rand;

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

    // •`‰æƒXƒŒƒbƒh‚©‚çŒÄ‚Î‚ê‚é
    void drawCallback();

    const InputState* getIngameInputs() const { return m_input_server->getInput(); }
    World* getWorld()   { return m_world; }
    SFMT* getRandom()   { return &m_rand; }
};

#define atomicGetWorld()            atomicGetGame()->getWorld()
#define atomicGetIngameInputs()     atomicGetGame()->getIngameInputs()
#define atomicGetRandom()           atomicGetGame()->getRandom()


} // namespace atomic
#endif __atomic_Game_AtomicGame__
