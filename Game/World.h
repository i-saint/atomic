#ifndef __atomic_Game_World_h__
#define __atomic_Game_World_h__

#include "Task.h"

namespace atomic {

class EntitySet;
class CollisionSet;
class SPHManager;


class World : boost::noncopyable
{
private:
    EntitySet       *m_entity_set;
    CollisionSet    *m_collision_set;
    SPHManager      *m_sph;

    Task_UpdateAsync<World>         *m_task_update_world;
    Task_UpdateAsync<EntitySet>     *m_task_update_entity;
    Task_UpdateAsync<CollisionSet>  *m_task_update_collision;
    Task_UpdateAsync<SPHManager>    *m_task_update_sph;

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
    void asyncupdate(float32 dt);
    void asyncupdateBegin(float32 dt);
    void asyncupdateEnd();
    void draw() const;

    uint32 getFrame() const         { return m_frame; }
    PerspectiveCamera* getCamera()  { return &m_camera; }
    SFMT* getRandom()               { return &m_rand; }

    EntitySet*  getEntitySet()      { return m_entity_set; }
    SPHManager* getFractionSet()   { return m_sph; }
};


#define atomicGetFrame()        atomicGetWorld()->getFrame()
#define atomicGetCamera()       atomicGetWorld()->getCamera()
#define atomicGetRandom()       atomicGetWorld()->getRandom()
#define atomicGenRandFloat()    atomicGetRandom()->genFloat32()
#define atomicGenRandVector2()  atomicGetRandom()->genVector2()
#define atomicGenRandVector3()  atomicGetRandom()->genVector3()
#define atomicGenRandVector4()  atomicGetRandom()->genVector4()

#define atomicGetEntitySet()    atomicGetWorld()->getEntitySet()
#define atomicGetEntity(id)     atomicGetEntitySet()->getEntity(id)
#define atomicCreateEntity(n)   atomicGetEntitySet()->createEntity<n>()
#define atomicDeleteEntity(o)   atomicGetEntitySet()->deleteEntity(o)

#define atomicGetSPHManager()   atomicGetWorld()->getFractionSet()


} // namespace atomic
#endif // __atomic_Game_World_h__
