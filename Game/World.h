#ifndef __atomic_Game_World_h__
#define __atomic_Game_World_h__

#include "Task.h"

namespace atomic {

class EntitySet;
class CollisionSet;
class SPHManager;
class VFXSet;


class World : public IAtomicGameModule
{
private:
    CollisionSet    *m_collision_set;
    SPHManager      *m_sph;
    EntitySet       *m_entity_set;
    VFXSet          *m_vfx;
    typedef stl::vector<IAtomicGameModule*> ModuleCont;
    ModuleCont      m_modules;

    typedef Task_UpdateAsync<IAtomicGameModule> ModuleUpdateTask;
    typedef stl::vector<ModuleUpdateTask*> ModuleUpdateTaskCont;
    ModuleUpdateTaskCont m_module_update_tasks;

    PerspectiveCamera m_camera;
    vec4 m_field_size;

    float32 m_frame;

public:
    World();
    ~World();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void asyncupdateBegin(float32 dt);
    void asyncupdateEnd();
    void draw();
    void frameEnd();

    PerspectiveCamera* getCamera() { return &m_camera; }
    const vec4& getFieldSize() const    { return m_field_size; }
    float32 getFrame() const            { return m_frame; }

    void setFieldSize(const vec4 &v)    { m_field_size=v; }

    EntitySet*      getEntitySet()      { return m_entity_set; }
    CollisionSet*   getCollisionSet()   { return m_collision_set; }
    SPHManager*     getFractionSet()    { return m_sph; }
};


#define atomicGetFrame()        atomicGetWorld()->getFrame()
#define atomicGetCamera()       atomicGetWorld()->getCamera()
#define atomicGenRandFloat()    atomicGetRandom()->genFloat32()

#define atomicGetEntitySet()    atomicGetWorld()->getEntitySet()
#define atomicGetEntity(id)     atomicGetEntitySet()->getEntity(id)
#define atomicCreateEntity(n)   atomicGetEntitySet()->createEntity<n>()
#define atomicDeleteEntity(o)   atomicGetEntitySet()->deleteEntity(o)

#define atomicGetCollisionSet()     atomicGetWorld()->getCollisionSet()
#define atomicCreateCollision(n)    atomicGetCollisionSet()->createEntity<n>()
#define atomicDeleteCollision(o)    atomicGetCollisionSet()->deleteEntity(o)
#define atomicGetCollision(h)       atomicGetCollisionSet()->getEntity(h)

#define atomicGetSPHManager()   atomicGetWorld()->getFractionSet()


} // namespace atomic
#endif // __atomic_Game_World_h__
