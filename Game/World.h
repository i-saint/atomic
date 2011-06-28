#ifndef __atomic_Game_World__
#define __atomic_Game_World__

class SFMT;

namespace atomic
{

class EnemySet;
class ForceSet;
class FractionSet;
class VFXSet;


class World
{
public:
    class Interframe
    {
    private:
        World* m_current_world;

    public:
        Interframe();
        void setCurrentWorld(World* w) { m_current_world=w; }
        World* getCurrentWorld() { return m_current_world; }
    };

private:
    static Interframe *s_interframe;

public:
    static void InitializeInterframe();
    static void FinalizeInterframe();
    static Interframe* getInterframe() { return s_interframe; }

private:
    FrameScopedAllocator *m_frame_alloc;
    FractionSet *m_fraction_set;
    World *m_prev, *m_next;

    SFMT m_rand;
    PerspectiveCamera m_camera;

public:
    World(World* prev);
    ~World();

    void update();
    void sync();
    void flushMessage();
    void processMessage();
    void draw();

    World* getPrev() { return m_prev; }
    World* getNext() { return m_next; }

    FractionSet* getFractions() { return m_fraction_set; }
    PerspectiveCamera* getCamera() { return &m_camera; }
    SFMT* getRandom() { return &m_rand; }
};


#define GetWorld() World::getInterframe()->getCurrentWorld()
#define GetPrevWorld() GetWorld()->getPrev()

#define GetFractions() GetWorld()->getFractions()
#define GetCamera() GetWorld()->getCamera()

#define GenFloatRand() GetWorld()->getRandom()->genFloat32()
#define GenVector2Rand() GetWorld()->getRandom()->genVector2()
#define GenVector3Rand() GetWorld()->getRandom()->genVector3()
#define GenVector4Rand() GetWorld()->getRandom()->genVector4()

class Task_WorldUpdate;
class Task_WorldDraw;


} // namespace atomic
#endif // __atomic_Game_World__
