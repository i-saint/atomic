#ifndef atomic_Game_World_h
#define atomic_Game_World_h

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
    SPHManager     *m_sph;
    EntitySet       *m_entity_set;
    VFXSet          *m_vfx;
    typedef stl::vector<IAtomicGameModule*> ModuleCont;
    ModuleCont      m_modules;

    typedef ist::Method<IAtomicGameModule, void, float32> ModuleAsyncupdateFunc;
    typedef ist::AsyncFunction<ModuleAsyncupdateFunc> ModuleAsyncupdateTask;
    typedef stl::vector<ModuleAsyncupdateTask> ModuleAsyncupdateTaskCont;
    ModuleAsyncupdateTaskCont m_module_asyncupdates;

    PerspectiveCamera m_camera_game;
    PerspectiveCamera m_camera_bg;
    FrustumPlanes m_frustum;
    vec4 m_field_size;

    uint32 m_frame;

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

    PerspectiveCamera* getGameCamera()  { return &m_camera_game; }
    PerspectiveCamera* getBGCamera()    { return &m_camera_bg; }
    const FrustumPlanes* getViewFrustum() const { return &m_frustum; }
    const vec4& getFieldSize() const    { return m_field_size; }
    uint32 getFrame() const             { return m_frame; }

    void setFieldSize(const vec4 &v)    { m_field_size=v; }

    EntitySet*      getEntitySet()      { return m_entity_set; }
    CollisionSet*   getCollisionSet()   { return m_collision_set; }
    SPHManager*    getFractionSet()    { return m_sph; }

    void handleEntitiesQuery(EntitiesQueryContext &ctx);
};


#define atomicGetFrame()        atomicGetWorld()->getFrame()
#define atomicGetGameCamera()   atomicGetWorld()->getGameCamera()
#define atomicGetBGCamera()     atomicGetWorld()->getBGCamera()
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
#endif // atomic_Game_World_h
