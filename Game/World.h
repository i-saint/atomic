#ifndef atm_Game_World_h
#define atm_Game_World_h

#include "Task.h"

namespace atm {

class EntityModule;
class BulletModule;
class CollisionModule;
class FluidModule;
class VFXModule;


class World : public IAtomicGameModule
{
public:
    World();
    ~World();

    void initialize();

    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void asyncupdateBegin(float32 dt);
    void asyncupdateEnd();
    void draw();
    void frameEnd();

    PerspectiveCamera* getGameCamera()          { return &m_camera_game; }
    PerspectiveCamera* getBGCamera()            { return &m_camera_bg; }
    const FrustumPlanes* getViewFrustum() const { return &m_frustum; }
    const vec4& getFieldSize() const            { return m_field_size; }

    void setFieldSize(const vec4 &v)    { m_field_size=v; }

    CollisionModule*    getCollisionModule(){ return m_collision_module; }
    FluidModule*        getFluidModule()    { return m_fluid_module; }
    EntityModule*       getEntityModule()   { return m_entity_module; }
    BulletModule*       getBulletModule()   { return m_bullet_module; }
    VFXModule*          getVFXModule()      { return m_vfx_module; }

    void handleEntitiesQuery(EntitiesQueryContext &ctx);

    istSerializeBlockDecl();

private:
    typedef ist::vector<IAtomicGameModule*> ModuleCont;

    CollisionModule *m_collision_module;
    FluidModule     *m_fluid_module;
    EntityModule    *m_entity_module;
    BulletModule    *m_bullet_module;
    VFXModule       *m_vfx_module;
    ModuleCont      m_modules;

    PerspectiveCamera m_camera_game;
    PerspectiveCamera m_camera_bg;
    FrustumPlanes m_frustum;
    vec4 m_field_size;

    // non serializable
    TaskGroup       m_asyncupdate;
};


#define atmGetGameCamera()      atmGetWorld()->getGameCamera()
#define atmGetBGCamera()        atmGetWorld()->getBGCamera()
#define atmGetViewFrustum()     atmGetWorld()->getViewFrustum()
#define atmGenRandFloat()       atmGetRandom()->genFloat32()

#define atmGetEntityModule()    atmGetWorld()->getEntityModule()
#define atmGetEntity(id)        atmGetEntityModule()->getEntity(id)
#define atmCreateEntity(C)      atmGetEntityModule()->createEntity(EC_##C)
#define atmDeleteEntity(o)      atmGetEntityModule()->deleteEntity(o)
#define atmEnumlateEntity(...)  atmGetEntityModule()->enumlateEntity(__VA_ARGS__)

#define atmGetCollisionModule() atmGetWorld()->getCollisionModule()
#define atmCreateCollision(n)   atmGetCollisionModule()->createEntity<n>()
#define atmDeleteCollision(o)   atmGetCollisionModule()->deleteEntity(o)
#define atmGetCollision(h)      atmGetCollisionModule()->getEntity(h)

#define atmGetFluidModule()     atmGetWorld()->getFluidModule()

#define atmGetBulletModule()    atmGetWorld()->getBulletModule()

} // namespace atm
#endif // atm_Game_World_h
