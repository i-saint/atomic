#ifndef __atomic_AtomicGame__
#define __atomic_AtomicGame__
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

    void update();
    void draw();
};

} // namespace atomic
#endif __atomic_AtomicGame__
