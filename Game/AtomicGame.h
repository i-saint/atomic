#ifndef __atomic_AtomicGame__
#define __atomic_AtomicGame__
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
    World *m_draw_target;
    uint32 m_current_world;

public:
    AtomicGame();
    ~AtomicGame();

    void update();
    void draw();
};

} // namespace atomic
#endif __atomic_AtomicGame__
