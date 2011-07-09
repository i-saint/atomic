#ifndef __atomic_Game_World__
#define __atomic_Game_World__

class SFMT;

namespace atomic
{

class EnemySet;
class ForceSet;
class FractionSet;
class VFXSet;
class World;
class Task_WorldBeforeDraw;
class Task_WorldAfterDraw;


class World : boost::noncopyable
{
public:
    class Interframe : boost::noncopyable
    {
    private:
        World* m_current_world;
        Task_WorldBeforeDraw *m_task_beforedraw;
        Task_WorldAfterDraw *m_task_afterdraw;

    public:
        Interframe();
        ~Interframe();
        void setCurrentWorld(World* w) { m_current_world=w; }
        World* getCurrentWorld() { return m_current_world; }

        Task_WorldBeforeDraw*    getTask_BeforeDraw() { return m_task_beforedraw; }
        Task_WorldAfterDraw*     getTask_AfterDraw() { return m_task_afterdraw; }
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

    void initialize(World* prev);

    void update();
    void draw();
    void sync();

    const World* getPrev() const { return m_prev; }

    FractionSet* getFractions() { return m_fraction_set; }
    PerspectiveCamera* getCamera() { return &m_camera; }
    SFMT* getRandom() { return &m_rand; }


public:
    void update_BeforeDraw();
    void update_AfterDraw();
};


#define atomicGetWorld()        World::getInterframe()->getCurrentWorld()
#define atomicGetPrevWorld()    atomicGetWorld()->getPrev()

#define atomicGetFractions()    atomicGetWorld()->getFractions()
#define atomicGetCamera()       atomicGetWorld()->getCamera()

#define atomicGetRandom()       atomicGetWorld()->getRandom()
#define atomicGenFloatRand()    atomicGetRandom()->genFloat32()
#define atomicGenVector2Rand()  atomicGetRandom()->genVector2()
#define atomicGenVector3Rand()  atomicGetRandom()->genVector3()
#define atomicGenVector4Rand()  atomicGetRandom()->genVector4()


} // namespace atomic
#endif // __atomic_Game_World__
