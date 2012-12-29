#ifndef __atomic_Game_World_h__
#define __atomic_Game_World_h__

#include "Task.h"

namespace atomic {

class EntitySet;
class CollisionSet;
class SPHManager2;
class VFXSet;


class World : public IAtomicGameModule
{
private:
    CollisionSet    *m_collision_set;
    SPHManager2     *m_sph;
    EntitySet       *m_entity_set;
    VFXSet          *m_vfx;
    typedef stl::vector<IAtomicGameModule*> ModuleCont;
    ModuleCont      m_modules;

    typedef ist::AsyncMethod<IAtomicGameModule, void, float> ModuleAsyncupdate;
    typedef stl::vector<ModuleAsyncupdate> ModuleAsyncupdateCont;
    ModuleAsyncupdateCont m_module_asyncupdates;

    PerspectiveCamera m_camera;
    FrustumPlanes m_frustum;
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

    PerspectiveCamera* getCamera()      { return &m_camera; }
    const FrustumPlanes* getViewFrustum() const { return &m_frustum; }
    const vec4& getFieldSize() const    { return m_field_size; }
    float32 getFrame() const            { return m_frame; }

    void setFieldSize(const vec4 &v)    { m_field_size=v; }

    EntitySet*      getEntitySet()      { return m_entity_set; }
    CollisionSet*   getCollisionSet()   { return m_collision_set; }
    SPHManager2*    getFractionSet()    { return m_sph; }
};


#define atomicGetFrame()        atomicGetWorld()->getFrame()
#define atomicGetCamera()       atomicGetWorld()->getCamera()
#define atomicGetViewFrustum()  atomicGetWorld()->getViewFrustum()
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
