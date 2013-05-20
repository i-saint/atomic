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
    float32 m_delta_fluid_damage;
    SE_CHANNEL m_explosion_channel;
    SE_RID m_explosion_se;
    vec4 m_light_color;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerializeBase(model)
        istSerializeBase(collision)
        istSerializeBase(bloodstain)
        istSerialize(m_state)
        istSerialize(m_st_frame)
        istSerialize(m_light_radius)
        istSerialize(m_delta_fluid_damage)
        istSerialize(m_explosion_channel)
        istSerialize(m_explosion_se)
        istSerialize(m_light_color)
        )

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

public:
    Enemy_Test() : m_state(ST_FADEIN), m_st_frame(0), m_light_radius(0.5f), m_delta_fluid_damage(0.0f)
        , m_explosion_channel(SE_CHANNEL3), m_explosion_se(SE_EXPLOSION3)
        , m_light_color(0.8f, 0.1f, 0.2f, 1.0f)
    {
        wdmScope( wdmString path = wdmFormat("Enemy/handle:0x%x", getHandle()) );
        wdmScope(super::addDebugNodes(path));
        wdmScope(transform::addDebugNodes(path));
        wdmScope(model::addDebugNodes(path));
        wdmAddNode(path+"/m_light_radius", &m_light_radius );
        wdmAddNode(path+"/m_light_color", &m_light_color, 0.0f, 1.0f );
    }

    ~Enemy_Test()
    {
        wdmEraseNode(wdmFormat("Enemy/handle:0x%x", getHandle()));
    }

    virtual void initialize()
    {
        super::initialize();
        setModel(PSET_CUBE_MEDIUM);
        setDiffuseColor(vec4(0.6f, 0.6f, 0.6f, 50.0f));
        setGlowColor(vec4(1.0f, 0.0f, 0.2f, 0.0f));
        setLife(100.0f);
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

        damage(m_delta_fluid_damage);
        m_delta_fluid_damage = 0.0f;

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
                return;
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
        vec4 diffuse = getDiffuseColor();
        vec4 glow = getGlowColor();
        vec4 light = m_light_color;
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
            inst.appear_radius = inst.elapsed * 0.004f;
            inst.translate = getTransform();
            atomicGetSPHRenderer()->addPSetInstance(getModel(), inst);
            atomicGetBloodstainRenderer()->addBloodstainParticles(getTransform(), getBloodStainParticles(), getNumBloodstainParticles());
        }
    }

    virtual void destroy()
    {
        setState(ST_FADEOUT);
        setRoutine(RCID_Null);
        atomicGetSPHManager()->addFluid(getModel(), getTransform());
        atomicPlaySE(m_explosion_channel, m_explosion_se, getPosition(), true);
    }

    virtual void eventFluid(const FluidMessage *m)
    {
        addBloodstain(getInverseTransform() * (vec4&)m->position);
        m_delta_fluid_damage += glm::length((const vec3&)m->velocity)*0.002f;
    }
};
atomicImplementEntity(Enemy_Test);
atomicExportClass(atomic::Enemy_Test);


class Routine_ChasePlayerRough : public IRoutine
{
typedef IRoutine super;
private:
    vec4 m_objective;
    int32 m_count;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_objective)
        istSerialize(m_count)
        )

public:
    Routine_ChasePlayerRough() : m_count(0)
    {
    }

    void update(float32 dt)
    {
    }
};
atomicExportClass(atomic::Routine_ChasePlayerRough);

} // namespace atomic

