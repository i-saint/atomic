#ifndef __atomic_Game_World_h__
#define __atomic_Game_World_h__

#include "Task.h"

namespace atomic {

class EntitySet;
class FractionSet;

class World;


class World : boost::noncopyable
{
private:
    typedef Task_UpdateAsync<World> UpdateAsyncTask;
    UpdateAsyncTask *m_task_updateasync;

    EntitySet   *m_entity_set;
    FractionSet *m_fraction_set;

    SFMT m_rand;
    PerspectiveCamera m_camera;

    uint32 m_frame;

public:
    World();
    ~World();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update(float32 dt);
    void draw() const;
    void sync() const;
    void updateAsync();

    uint32 getFrame() const         { return m_frame; }
    PerspectiveCamera* getCamera()  { return &m_camera; }
    SFMT* getRandom()               { return &m_rand; }

    EntitySet*  getEntitySet()      { return m_entity_set; }
    FractionSet* getFractionSet()   { return m_fraction_set; }
};


#define atomicGetFrame()        atomicGetWorld()->getFrame()
#define atomicGetCamera()       atomicGetWorld()->getCamera()
#define atomicGetRandom()       atomicGetWorld()->getRandom()
#define atomicGenRandFloat()    atomicGetRandom()->genFloat32()
#define atomicGenRandVector2()  atomicGetRandom()->genVector2()
#define atomicGenRandVector3()  atomicGetRandom()->genVector3()
#define atomicGenRandVector4()  atomicGetRandom()->genVector4()

#define atomicGetEntitySet()    atomicGetWorld()->getEntitySet()
#define atomicGetFractionSet()  atomicGetWorld()->getFractionSet()


} // namespace atomic
#endif // __atomic_Game_World_h__
