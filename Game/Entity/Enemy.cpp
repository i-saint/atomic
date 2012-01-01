#include "stdafx.h"
#include "types.h"
#include "Util.h"
#include "Sound/AtomicSound.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "Game/Collision.h"
#include "Game/Message.h"
#include "Routine.h"
#include "Enemy.h"

namespace atomic {


template<class CollisonType>
class Enemy_Test
    : public Breakable
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
    , public Attr_ParticleSet
    , public CollisonType
{
typedef Breakable super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
typedef Attr_ParticleSet model;
typedef CollisonType collision;
private:
    enum STATE {
        ST_FADEIN,
        ST_ACTIVE,
        ST_FADEOUT,
    };

    static const int FADEIN_TIME = 180;
    static const int FADEOUT_TIME = 60;
    STATE m_state;
    int32 m_st_frame;
    SE_CHANNEL m_explosion_channel;
    SE_RID m_explosion_se;

public:
    Enemy_Test() : m_state(ST_FADEIN), m_st_frame(0)
        , m_explosion_channel(SE_CHANNEL3), m_explosion_se(SE_EXPLOSION3)
    {
    }

    virtual void initialize()
    {
        super::initialize();
        setModel(PSET_CUBE_MEDIUM);
        setDiffuseColor(vec4(0.6f, 0.6f, 0.6f, 1.0f));
        setGlowColor(vec4(1.0f, 0.0f, 0.2f, 1.0f));
        setHealth(100.0f);
        collision::initializeCollision(getHandle());
    }

    void setState(STATE s) { m_state=s; m_st_frame=0; }
    STATE getState() const { return m_state; }

    void setExplosionSE(SE_RID v)           { m_explosion_se=v; }
    void setExplosionChannel(SE_CHANNEL v)  { m_explosion_channel=v; }

    virtual void update(float32 dt)
    {
        super::update(dt);
        transform::update(dt);

        setTransform(computeMatrix());

        ++m_st_frame;
        float32 rigid_scale = 1.0f;
        if(getState()==ST_FADEIN) {
            rigid_scale = ((float32)m_st_frame / FADEIN_TIME);
            if(m_st_frame==FADEIN_TIME) {
                setState(ST_ACTIVE);
            }
        }
        if(getState()!=ST_FADEOUT) {
            collision::updateCollision(getModel(), getTransform(), rigid_scale);
        }
        if(getState()==ST_FADEOUT) {
            if(m_st_frame==2) { // 1 �t���[���R���W�������c���ăp�[�e�B�N���𔚎U������
                collision::finalizeCollision();
            }
            if(m_st_frame==FADEOUT_TIME) {
                atomicDeleteEntity(getHandle());
            }
        }
    }

    virtual void updateRoutine(float32 dt)
    {
        if(getState()==ST_ACTIVE) {
            IRoutine *routine = getRoutine();
            if(routine) { routine->update(dt); }
        }
    }

    virtual void asyncupdateRoutine(float32 dt)
    {
        if(getState()==ST_ACTIVE) {
            IRoutine *routine = getRoutine();
            if(routine) { routine->asyncupdate(dt); }
        }
    }

    virtual void draw()
    {
        vec4 diffuse = getDiffuseColor();
        vec4 glow = getGlowColor();
        vec4 light = vec4(0.8f, 0.1f, 0.2f, 1.0f);
        if(getState()==ST_FADEIN) {
            float32 s = (float32)m_st_frame / FADEIN_TIME;
            diffuse *= std::min<float32>(s*2.0f, 1.0f);
            glow    *= std::max<float32>(s*2.0f-1.0f, 0.0f);
            light   *= s;
        }
        else if(getState()==ST_FADEOUT) {
            float32 s = 1.0f - ((float32)m_st_frame / FADEOUT_TIME);
            light   *= s;
        }

        {
            PointLight l;
            l.setPosition(getPosition() + vec4(0.0f, 0.0f, 0.2f, 1.0f));
            l.setColor(light);
            l.setRadius(1.0f);
            atomicGetPointLights()->addInstance(l);
        }
        if(m_state!=ST_FADEOUT) {
            atomicGetSPHRenderer()->addPSetInstance(getModel(), getTransform(), diffuse, glow, getFlashColor());
        }
    }

    virtual void destroy()
    {
        setState(ST_FADEOUT);
        setRoutine(ROUTINE_NULL);
        atomicGetSPHManager()->addFluid(getModel(), getTransform());
        atomicPlaySE(SE_CHANNEL3, m_explosion_se, getPosition(), true);
    }

    virtual bool call(uint32 call_id, const variant &v)
    {
        switch(call_id) {
        DEFINE_ECALL1(setExplosionSE, SE_RID);
        DEFINE_ECALL1(setExplosionChannel, SE_CHANNEL);
        default: return super::call(call_id, v) || transform::call(call_id, v) || model::call(call_id, v);
        }
    }

    virtual bool query(uint32 query_id, variant &v) const
    {
        return super::query(query_id, v) || transform::query(query_id, v) || model::query(query_id, v);
    }
};

class Enemy_CubeBasic : public Enemy_Test<Attr_CubeCollision>
{
};

class Enemy_SphereBasic : public Enemy_Test<Attr_SphereCollision>
{
};


class Routine_ChasePlayerRough : public IRoutine
{
private:
    vec4 m_objective;
    int32 m_count;

public:
    Routine_ChasePlayerRough() : m_count(0)
    {
    }

    void update(float32 dt)
    {

    }
};


atomicImplementEntity(Enemy_CubeBasic, ECID_ENEMY, ESID_ENEMY_CUBE);
atomicImplementEntity(Enemy_SphereBasic, ECID_ENEMY, ESID_ENEMY_SPHERE);

} // namespace atomic


istImplementClassInfo(atomic::Enemy_CubeBasic);
istImplementClassInfo(atomic::Enemy_SphereBasic);