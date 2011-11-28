#ifndef __atomic_Game_World_h__
#define __atomic_Game_World_h__

namespace atomic {

class CharacterSet;
class BulletSet;
class ForceSet;
class FractionSet;
class VFXSet;

class World;
class Task_WorlUpdateAsync;


class World : boost::noncopyable
{
private:
    Task_WorlUpdateAsync *m_task_updateasync;

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
    void updateAsync();

    uint32 getFrame() const { return m_frame; }
    PerspectiveCamera* getCamera() { return &m_camera; }
    SFMT* getRandom() { return &m_rand; }

    FractionSet* getFractions() { return m_fraction_set; }

};


#define atomicGetFrame()        atomicGetWorld()->getFrame()
#define atomicGetCamera()       atomicGetWorld()->getCamera()
#define atomicGetRandom()       atomicGetWorld()->getRandom()
#define atomicGenRandFloat()    atomicGetRandom()->genFloat32()
#define atomicGenRandVector2()  atomicGetRandom()->genVector2()
#define atomicGenRandVector3()  atomicGetRandom()->genVector3()
#define atomicGenRandVector4()  atomicGetRandom()->genVector4()

#define atomicGetFractions()    atomicGetWorld()->getFractions()


} // namespace atomic
#endif // __atomic_Game_World_h__
