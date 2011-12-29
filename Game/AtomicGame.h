#ifndef __atomic_Game_AtomicGame__
#define __atomic_Game_AtomicGame__
namespace atomic {

class World;
class AtomicRenderer;

class AtomicGame
{
private:
    World *m_world;

public:
    AtomicGame();
    ~AtomicGame();

    void update(float32 dt);
    void asyncupdateBegin(float32 dt);
    void asyncupdateEnd();
    void draw();

    // •`‰æƒXƒŒƒbƒh‚©‚çŒÄ‚Î‚ê‚é
    void drawCallback();


    World* getWorld() { return m_world; }
};

#define atomicGetWorld()    atomicGetGame()->getWorld()


} // namespace atomic
#endif __atomic_Game_AtomicGame__
