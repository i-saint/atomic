#ifndef __atomic_Game_AtomicGame__
#define __atomic_Game_AtomicGame__
#include "Input.h"

namespace atomic {

class World;
class AtomicRenderer;

class AtomicGame
{
private:
    IInputServer  *m_input_server;
    World               *m_world;

public:
    AtomicGame();
    ~AtomicGame();

    bool readReplayFromFile(const char *path);

    void update(float32 dt);
    void asyncupdateBegin(float32 dt);
    void asyncupdateEnd();
    void draw();

    // •`‰æƒXƒŒƒbƒh‚©‚çŒÄ‚Î‚ê‚é
    void drawCallback();

    World* getWorld() { return m_world; }
    const InputState* getIngameInputs() const { return m_input_server->getInput(); }
};

#define atomicGetWorld()            atomicGetGame()->getWorld()
#define atomicGetIngameInputs()     atomicGetGame()->getIngameInputs()


} // namespace atomic
#endif __atomic_Game_AtomicGame__
