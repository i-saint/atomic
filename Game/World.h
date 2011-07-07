#ifndef __atomic_Game_World__
#define __atomic_Game_World__

class SFMT;

namespace atomic
{

class EnemySet;
class ForceSet;
class FractionSet;
class VFXSet;


class World : boost::noncopyable
{
public:
    class Interframe : boost::noncopyable
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
    static void initializeInterframe();
    static void finalizeInterframe();
    static Interframe* getInterframe() { return s_interframe; }

private:
    World *m_prev;
    FractionSet *m_fraction_set;

    SFMT m_rand;
    PerspectiveCamera m_camera;

public:
    World();
    ~World();

    void initialize(World* prev, FrameAllocator& alloc);

    void update();
    void sync();
    void flushMessage();
    void processMessage();
    void draw();

    World* getPrev() { return m_prev; }

    FractionSet* getFractions() { return m_fraction_set; }
    PerspectiveCamera* getCamera() { return &m_camera; }
    SFMT* getRandom() { return &m_rand; }
};


#define GetWorld()          World::getInterframe()->getCurrentWorld()
#define GetPrevWorld()      GetWorld()->getPrev()

#define GetFractions()      GetWorld()->getFractions()
#define GetCamera()         GetWorld()->getCamera()

#define GetRandom()         GetWorld()->getRandom()
#define GenFloatRand()      GetRandom()->genFloat32()
#define GenVector2Rand()    GetRandom()->genVector2()
#define GenVector3Rand()    GetRandom()->genVector3()
#define GenVector4Rand()    GetRandom()->genVector4()

class Task_WorldUpdate;
class Task_WorldDraw;


} // namespace atomic
#endif // __atomic_Game_World__
