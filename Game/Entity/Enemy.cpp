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


class Enemy_Test
    : public Breakable
    , public TAttr_TransformMatrixI< TAttr_RotateSpeed<Attr_DoubleAxisRotation> >
    , public Attr_ParticleSet
    , public Attr_Collision
    , public Attr_Bloodstain
{
typedef Enemy_Test this_t;
typedef Breakable super;
typedef TAttr_TransformMatrixI< TAttr_RotateSpeed<Attr_DoubleAxisRotation> > transform;
typedef Attr_ParticleSet    model;
typedef Attr_Collision      collision;
typedef Attr_Bloodstain     bloodstain;
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
    float32 m_light_radius;
    SE_CHANNEL m_explosion_channel;
    SE_RID m_explosion_se;

public:
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setLightRadius)
        atomicECall(setExplosionSE)
        atomicECall(setExplosionChannel)
        )
        atomicECallSuper(super)
        atomicECallSuper(transform)
        atomicECallSuper(model)
        atomicECallSuper(collision)
    )
    atomicEQueryBlock(
        atomicEQuerySuper(super)
        atomicEQuerySuper(transform)
        atomicEQuerySuper(model)
        atomicEQuerySuper(collision)
    )

public:
    Enemy_Test() : m_state(ST_FADEIN), m_st_frame(0), m_light_radius(0.5f)
        , m_explosion_channel(SE_CHANNEL3), m_explosion_se(SE_EXPLOSION3)
    {
    }

    virtual void initialize()
    {
        super::initialize();
        setModel(PSET_CUBE_MEDIUM);
        setDiffuseColor(vec4(0.6f, 0.6f, 0.6f, 50.0f));
        setGlowColor(vec4(1.0f, 0.0f, 0.2f, 0.0f));
        setHealth(100.0f);
        collision::initializeCollision(getHandle());
    }

    void setState(STATE s) { m_state=s; m_st_frame=0; }
    STATE getState() const { return m_state; }

    void setLightRadius(float32 v)          { m_light_radius=v; }
    void setExplosionSE(SE_RID v)           { m_explosion_se=v; }
    void setExplosionChannel(SE_CHANNEL v)  { m_explosion_channel=v; }

    virtual void update(float32 dt)
    {
        super::update(dt);

        ++m_st_frame;
        float32 rigid_scale = 1.0f;
        if(getState()==ST_FADEIN) {
            rigid_scale = ((float32)m_st_frame / FADEIN_TIME);
            if(m_st_frame==FADEIN_TIME) {
                setState(ST_ACTIVE);
            }
        }
        if(getState()==ST_FADEOUT) {
            if(m_st_frame==2) { // 1 フレームコリジョンを残してパーティクルを爆散させる
                collision::finalizeCollision();
            }
            if(m_st_frame==FADEOUT_TIME) {
                atomicDeleteEntity(getHandle());
            }
        }
    }

    virtual void asyncupdate(float32 dt)
    {
        super::asyncupdate(dt);
        transform::updateRotate(dt);
        transform::updateTransformMatrix();
        bloodstain::updateBloodstain();

        float32 rigid_scale = 1.0f;
        if(getState()==ST_FADEIN) {
            rigid_scale = ((float32)m_st_frame / FADEIN_TIME);
        }
        if(getState()!=ST_FADEOUT) {
            collision::updateCollisionByParticleSet(getModel(), getTransform(), rigid_scale);
        }
    }

    virtual void updateRoutine(float32 dt)
    {
        if(getState()==ST_ACTIVE) {
            if(IRoutine *routine = getRoutine()) {
                routine->update(dt);
            }
        }
    }

    virtual void asyncupdateRoutine(float32 dt)
    {
        if(getState()==ST_ACTIVE) {
            if(IRoutine *routine = getRoutine()) {
                routine->asyncupdate(dt);
            }
        }
    }

    virtual void draw()
    {
        //return;

        vec4 diffuse = getDiffuseColor();
        vec4 glow = getGlowColor();
        vec4 light = vec4(0.8f, 0.1f, 0.2f, 1.0f);
        if(getState()==ST_FADEIN) {
            float32 s   = (float32)m_st_frame / FADEIN_TIME;
            float shininess = diffuse.w;
            diffuse     *= stl::min<float32>(s*2.0f, 1.0f);
            diffuse.w   = shininess;
            glow        *= stl::max<float32>(s*2.0f-1.0f, 0.0f);
            light       *= s;
        }
        else if(getState()==ST_FADEOUT) {
            float32 s = 1.0f - ((float32)m_st_frame / FADEOUT_TIME);
            light   *= s;
        }

        if(m_light_radius > 0.0f) {
            PointLight l;
            l.setPosition(getPosition() + vec4(0.0f, 0.0f, m_light_radius*0.5f, 1.0f));
            l.setColor(light);
            l.setRadius(m_light_radius);
            atomicGetLights()->addLight(l);
        }
        if(m_state!=ST_FADEOUT) {
            PSetInstance inst;
            inst.diffuse = diffuse;
            inst.glow = glow;
            inst.flash = getFlashColor();
            inst.elapsed = (float32)getPastFrame();
            atomicGetSPHRenderer()->addPSetInstance(getModel(), getTransform(), inst);
            atomicGetBloodstainRenderer()->addBloodstainParticles(getTransform(), getBloodStainParticles(), getNumBloodstainParticles());
        }
    }

    virtual void destroy()
    {
        setState(ST_FADEOUT);
        setRoutine(ROUTINE_NULL);
        atomicGetSPHManager()->addFluid(getModel(), getTransform());
        atomicPlaySE(SE_CHANNEL3, m_explosion_se, getPosition(), true);
    }

    virtual void eventFluid(const FluidMessage *m)
    {
        addBloodstain(getInverseTransform() * (vec4&)m->position);
        damage(glm::length((const vec3&)m->velocity)*0.002f);
    }
};
atomicImplementEntity(Enemy_Test, ECID_Enemy);


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

} // namespace atomic


istImplementClassInfo(atomic::Enemy_Test);
