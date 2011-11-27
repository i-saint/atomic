#ifndef __atomic_Game_World_h__
#define __atomic_Game_World_h__

namespace atomic {

class CharacterSet;
class BulletSet;
class ForceSet;
class FractionSet;
class VFXSet;

class World;
class Task_WorldBeforeDraw;
class Task_WorldAfterDraw;
class Task_WorldCopy;


class World : boost::noncopyable
{
public:
    class Interframe : boost::noncopyable
    {
    private:
        World* m_current_world;
        Task_WorldBeforeDraw *m_task_beforedraw;
        Task_WorldAfterDraw *m_task_afterdraw;
        Task_WorldCopy *m_task_copy;

    public:
        Interframe();
        ~Interframe();
        void setCurrentWorld(World* w) { m_current_world=w; }
        World* getCurrentWorld() { return m_current_world; }

        Task_WorldBeforeDraw*   getTask_BeforeDraw() { return m_task_beforedraw; }
        Task_WorldAfterDraw*    getTask_AfterDraw() { return m_task_afterdraw; }
        Task_WorldCopy*         getTask_Copy() { return m_task_copy; }
    };

private:
    static Interframe *s_interframe;

public:
    static void initializeInterframe();
    static void finalizeInterframe();
    static Interframe* getInterframe() { return s_interframe; }

private:
    const World *m_prev;
    World *m_next;

    FractionSet *m_fraction_set;
    BulletSet *m_bullet_set;

    SFMT m_rand;
    PerspectiveCamera m_camera;

    uint32 m_frame;

public:
    World();
    ~World();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void draw() const;
    void sync() const;

    void setNext(World *next);
    World* getNext() { return m_next; }
    const World* getPrev() const { return m_prev; }

    uint32 getFrame() const { return m_frame; }
    FractionSet* getFractions() { return m_fraction_set; }
    PerspectiveCamera* getCamera() { return &m_camera; }
    SFMT* getRandom() { return &m_rand; }


public:
    void taskBeforeDraw();
    void taskAfterDraw();
    void taskCopy(World *dst) const;
};


#define atomicGetWorld()        World::getInterframe()->getCurrentWorld()
#define atomicGetPrevWorld()    atomicGetWorld()->getPrev()
#define atomicGetFrame()        atomicGetWorld()->getFrame()

#define atomicGetFractions()    atomicGetWorld()->getFractions()
#define atomicGetCamera()       atomicGetWorld()->getCamera()

#define atomicGetRandom()       atomicGetWorld()->getRandom()
#define atomicGenRandFloat()    atomicGetRandom()->genFloat32()
#define atomicGenRandVector2()  atomicGetRandom()->genVector2()
#define atomicGenRandVector3()  atomicGetRandom()->genVector3()
#define atomicGenRandVector4()  atomicGetRandom()->genVector4()


} // namespace atomic
#endif // __atomic_Game_World_h__
