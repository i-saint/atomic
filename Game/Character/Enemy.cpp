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

namespace atomic {


class Enemy_Base : public IEntity
{
private:
    mat4        m_transform;
    IRoutine    *m_routine;
    float32     m_health;

public:
    Enemy_Base() : m_transform(), m_routine(NULL), m_health(1.0f) {}

    float32     getHealth() const   { return m_health; }
    const mat4& getTransform() const{ return m_transform; }
    IRoutine*   getRoutine()        { return m_routine; }

    void setHealth(float32 v)        { m_health=v; }
    void setTransform(const mat4& v){ m_transform=v; }
    void setRoutine(IRoutine *v)    { m_routine=v; }

    virtual void update(float32 dt)
    {
        if(m_routine) { m_routine->update(dt); }
    }

    virtual void updateAsync(float32 dt)
    {
        if(m_routine) { m_routine->updateAsync(dt); }
    }

    virtual bool call(uint32 call_id, const variant &v)
    {
        switch(call_id) {
            DEFINE_ECALL1(setHealth, float32);
        }
        return false;
    }

    virtual bool query(uint32 query_id, variant &v) const
    {
        switch(query_id) {
            DEFINE_EQUERY(getHealth);
        }
        return false;
    }

    virtual void onDamage(const DamageMessage &m)
    {

    }
};


class Enemy_Cube
    : public Enemy_Base
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
{
typedef Enemy_Base super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
private:

public:
    virtual void update(float32 dt)
    {
        super::update(dt);
        transform::update(dt);
        setTransform(computeMatrix());
        atomicGetFractionSet()->addRigid(CB_CLASS_CUBE, getHandle(), getTransform());
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
    : public Enemy_Base
    , public TAttr_RotateSpeed<Attr_DoubleAxisRotation>
{
typedef Enemy_Base super;
typedef TAttr_RotateSpeed<Attr_DoubleAxisRotation> transform;
private:

public:
    virtual void update(float32 dt)
    {
        super::update(dt);
        transform::update(dt);
        setTransform(computeMatrix());
        atomicGetFractionSet()->addRigid(CB_CLASS_SPHERE, getHandle(), getTransform());
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


atomicImplementEntity(Enemy_Cube, ECID_ENEMY, ESID_ENEMY_CUBE);
atomicImplementEntity(Enemy_Sphere, ECID_ENEMY, ESID_ENEMY_SPHERE);

} // namespace atomic


istImplementClassInfo(atomic::Enemy_Cube);
istImplementClassInfo(atomic::Enemy_Sphere);
