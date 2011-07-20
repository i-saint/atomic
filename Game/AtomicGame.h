#ifndef __atomic_AtomicGame_h__
#define __atomic_AtomicGame_h__
namespace atomic {

class World;
class AtomicRenderer;

class AtomicGame
{
public:
    static const uint32 MAX_WORLDS = 3;

private:
    typedef stl::vector<World*> WorldCont;
    WorldCont m_worlds;
    World *m_current;
    World *m_prev;
    World *m_draw_target;
    uint32 m_world_index;

public:
    AtomicGame();
    ~AtomicGame();

    void update();
    void draw();

    // •`‰æƒXƒŒƒbƒh‚©‚çŒÄ‚Î‚ê‚é
    void drawCallback();
};

} // namespace atomic
#endif __atomic_AtomicGame_h__
