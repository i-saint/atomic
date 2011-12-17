#include "stdafx.h"
#include "types.h"
#include "Game/AtomicGame.h"
#include "Graphics/ResourceManager.h"
#include "Game/Entity.h"
#include "Enemy.h"

namespace atomic {


class Enemy_Base : public IEntity
{
private:
    mat4        m_transform;
    IRoutine    *m_routine;
    uint32      m_handle;
    float32     m_health;

public:
    Enemy_Base() : m_transform(), m_routine(NULL), m_handle(0), m_health(1.0f) {}

    float32     getHealth() const   { return m_health; }
    const mat4& getTransform() const{ return m_transform; }
    IRoutine*   getRoutine()        { return m_routine; }

    void setHelth(float32 v)        { m_health=v; }
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
};


class Enemy_Cube : public Enemy_Base, public Attr_DoubleAxisRotation
{
    typedef Enemy_Base super;
    typedef Attr_DoubleAxisRotation transform;
private:
public:
    virtual void updateAsync(float32 dt)
    {
        super::updateAsync(dt);
        setTransform(computeMatrix());
    }

    virtual void draw()
    {

    }
};

class Enemy_Sphere : public Enemy_Base, public Attr_DoubleAxisRotation
{
    typedef Enemy_Base super;
    typedef Attr_DoubleAxisRotation transform;
private:
public:
    virtual void updateAsync(float32 dt)
    {
        super::updateAsync(dt);
        setTransform(computeMatrix());
    }

    virtual void draw()
    {

    }
};


atomicImplementEntity(Enemy_Cube, ECID_ENEMY, ESID_ENEMY_CUBE);
atomicImplementEntity(Enemy_Sphere, ECID_ENEMY, ESID_ENEMY_SPHERE);

} // namespace atomic


istImplementClassInfo(atomic::Enemy_Cube);
istImplementClassInfo(atomic::Enemy_Sphere);
