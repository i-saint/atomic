#include "stdafx.h"
#include "types.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "Game/Message.h"
#include "Enemy.h"
#include "GPGPU/SPH.cuh"
#include "Util.h"

namespace atomic {

    class Attr_ParticleSet
    {
    private:
        vec4 m_diffuse_color;
        vec4 m_glow_color;
        PSET_RID m_psetid;

    public:
        Attr_ParticleSet() : m_psetid(PSET_CUBE_SMALL)
        {}

        void setDiffuseColor(const vec4 &v) { m_diffuse_color=v; }
        void setGlowColor(const vec4 &v)    { m_glow_color=v; }
        void setModel(PSET_RID v)           { m_psetid=v; }
        const vec4& getDiffuseColor() const { return m_diffuse_color; }
        const vec4& getGlowColor() const    { return m_glow_color; }
        PSET_RID getModel() const           { return m_psetid; }

        bool call(uint32 call_id, const variant &v)
        {
            switch(call_id) {
                DEFINE_ECALL1(setDiffuseColor, vec4);
                DEFINE_ECALL1(setGlowColor, vec4);
                DEFINE_ECALL1(setModel, PSET_RID);
            }
            return false;
        }

        bool query(uint32 query_id, variant &v) const
        {
            switch(query_id) {
                DEFINE_EQUERY(getDiffuseColor);
                DEFINE_EQUERY(getGlowColor);
                DEFINE_EQUERY(getModel);
            }
            return false;
        }

    };


class Enemy_CubeBasic
    : public Breakable
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
    , public Attr_ParticleSet
{
typedef Breakable super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
typedef Attr_ParticleSet model;
private:
    sphRigidBox m_rigid;

public:
    Enemy_CubeBasic()
    {
        setModel(PSET_CUBE_MEDIUM);
        setDiffuseColor(vec4(0.6f, 0.6f, 0.6f, 1.0f));
        setGlowColor(vec4(1.0f, 0.0f, 0.2f, 1.0f));
        setHealth(100.0f);
    }

    virtual void update(float32 dt)
    {
        super::update(dt);
        transform::update(dt);

        setTransform(computeMatrix());
        CreateRigidBox(m_rigid, getHandle(), getTransform(), (vec4&)atomicGetRigidInfo(getModel())->box_size * getScale());
        atomicGetSPHManager()->addRigidBox(m_rigid);
    }

    virtual void draw()
    {
        {
            PointLight light;
            light.position  = getPosition() + vec4(0.0f, 0.0f, 0.1f, 1.0f);
            light.color     = vec4(1.0f, 0.1f, 0.2f, 1.0f);
            atomicGetPointLights()->addInstance(light);
        }
        {
            atomicGetSPHRenderer()->addRigidInstance(getModel(), getTransform(), getDiffuseColor(), getGlowColor(), getFlashColor());
        }
    }

    virtual void destroy()
    {
        atomicGetSPHManager()->addFluidParticles(getModel(), getTransform());
        super::destroy();
    }

    bool call(uint32 call_id, const variant &v)
    {
        return super::call(call_id, v) || transform::call(call_id, v) || model::call(call_id, v);
    }

    bool query(uint32 query_id, variant &v) const
    {
        return super::query(query_id, v) || transform::query(query_id, v) || model::query(query_id, v);
    }
};

class Enemy_SphereBasic
    : public Breakable
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
    , public Attr_ParticleSet
{
typedef Breakable super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
typedef Attr_ParticleSet model;
private:
    sphRigidSphere m_rigid;

public:
    Enemy_SphereBasic()
    {
        setModel(PSET_SPHERE_LARGE);
        setDiffuseColor(vec4(0.6f, 0.6f, 0.6f, 1.0f));
        setGlowColor(vec4(1.0f, 0.0f, 0.2f, 1.0f));
        setHealth(100.0f);
    }

    virtual void update(float32 dt)
    {
        super::update(dt);
        transform::update(dt);

        setTransform(computeMatrix());
        CreateRigidSphere(m_rigid, getHandle(), getPosition(), atomicGetRigidInfo(getModel())->sphere_radius*getScale().x);
        atomicGetSPHManager()->addRigidSphere(m_rigid);
    }

    virtual void draw()
    {
        {
            PointLight light;
            light.position  = getPosition() + vec4(0.0f, 0.0f, 0.1f, 1.0f);
            light.color     = vec4(1.0f, 0.1f, 0.2f, 1.0f);
            atomicGetPointLights()->addInstance(light);
        }
        {
            atomicGetSPHRenderer()->addRigidInstance(getModel(), getTransform(), getDiffuseColor(), getGlowColor(), getFlashColor());
        }
    }

    virtual void destroy()
    {
        atomicGetSPHManager()->addFluidParticles(getModel(), getTransform());
        super::destroy();
    }

    bool call(uint32 call_id, const variant &v)
    {
        return super::call(call_id, v) || transform::call(call_id, v) || model::call(call_id, v);
    }

    bool query(uint32 query_id, variant &v) const
    {
        return super::query(query_id, v) || transform::query(query_id, v) || model::query(query_id, v);
    }
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
