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



class Enemy_Cube
    : public Breakable
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
{
typedef Breakable super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
private:
    static const PSET_RID rigid_class = PSET_CUBE_MEDIUM;
    sphRigidBox m_rigid;

public:
    Enemy_Cube()
    {
        setHealth(2000.0f);
    }

    virtual void update(float32 dt)
    {
        super::update(dt);
        transform::update(dt);

        setTransform(computeMatrix());
        CreateRigidBox(m_rigid, getHandle(), getTransform(), (vec4&)atomicGetRigidInfo(rigid_class)->box_size * getScale());
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
            atomicGetSPHRenderer()->addRigidInstance(rigid_class, getTransform(), vec4(0.6f, 0.6f, 0.6f, 1.0f), vec4(1.0f, 0.0f, 0.2f, 1.0f));
        }
    }

    bool call(uint32 call_id, const variant &v)
    {
        return super::call(call_id, v) || transform::call(call_id, v);
    }

    bool query(uint32 query_id, variant &v) const
    {
        return super::query(query_id, v) || transform::query(query_id, v);
    }
};

class Enemy_Sphere
    : public Breakable
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
{
typedef Breakable super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
private:
    static const PSET_RID rigid_class = PSET_SPHERE_LARGE;
    sphRigidSphere m_rigid;

public:
    Enemy_Sphere()
    {
        setHealth(1000.0f);
    }

    virtual void update(float32 dt)
    {
        super::update(dt);
        transform::update(dt);

        setTransform(computeMatrix());
        CreateRigidSphere(m_rigid, getHandle(), getPosition(), atomicGetRigidInfo(rigid_class)->sphere_radius*getScale().x);
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
            atomicGetSPHRenderer()->addRigidInstance(rigid_class, getTransform(), vec4(0.6f, 0.6f, 0.6f, 1.0f), vec4(1.0f, 0.0f, 0.2f, 1.0f));
        }
    }

    bool call(uint32 call_id, const variant &v)
    {
        return super::call(call_id, v) || transform::call(call_id, v);
    }

    bool query(uint32 query_id, variant &v) const
    {
        return super::query(query_id, v) || transform::query(query_id, v);
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


atomicImplementEntity(Enemy_Cube, ECID_ENEMY, ESID_ENEMY_CUBE);
atomicImplementEntity(Enemy_Sphere, ECID_ENEMY, ESID_ENEMY_SPHERE);

} // namespace atomic


istImplementClassInfo(atomic::Enemy_Cube);
istImplementClassInfo(atomic::Enemy_Sphere);
