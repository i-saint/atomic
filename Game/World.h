#ifndef atm_Game_World_h
#define atm_Game_World_h

#include "Task.h"

namespace atm {

class EntitySet;
class CollisionSet;
class SPHManager;
class VFXSet;


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

    PerspectiveCamera* getGameCamera()  { return &m_camera_game; }
    PerspectiveCamera* getBGCamera()    { return &m_camera_bg; }
    const FrustumPlanes* getViewFrustum() const { return &m_frustum; }
    const vec4& getFieldSize() const    { return m_field_size; }

    void setFieldSize(const vec4 &v)    { m_field_size=v; }

    EntitySet*      getEntitySet()      { return m_entity_set; }
    CollisionSet*   getCollisionSet()   { return m_collision_set; }
    SPHManager*    getFractionSet()    { return m_sph; }

    void handleEntitiesQuery(EntitiesQueryContext &ctx);

    istSerializeBlockDecl();

private:
    typedef ist::vector<IAtomicGameModule*> ModuleCont;

    CollisionSet    *m_collision_set;
    SPHManager      *m_sph;
    EntitySet       *m_entity_set;
    VFXSet          *m_vfx;
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

#define atmGetEntitySet()       atmGetWorld()->getEntitySet()
#define atmGetEntity(id)        atmGetEntitySet()->getEntity(id)
#define atmCreateEntity(C)      atmGetEntitySet()->createEntity(EC_##C)
#define atmDeleteEntity(o)      atmGetEntitySet()->deleteEntity(o)
#define atmEnumlateEntity(...)  atmGetEntitySet()->enumlateEntity(__VA_ARGS__)

#define atmGetCollisionSet()    atmGetWorld()->getCollisionSet()
#define atmCreateCollision(n)   atmGetCollisionSet()->createEntity<n>()
#define atmDeleteCollision(o)   atmGetCollisionSet()->deleteEntity(o)
#define atmGetCollision(h)      atmGetCollisionSet()->getEntity(h)

#define atmGetSPHManager()      atmGetWorld()->getFractionSet()


} // namespace atm
#endif // atm_Game_World_h
