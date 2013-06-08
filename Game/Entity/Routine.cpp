#include "stdafx.h"
#include "types.h"
#include "Graphics/ResourceManager.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/Collision.h"
#include "Game/Entity.h"
#include "Game/EntityQuery.h"
#include "Game/Entity/Attributes.h"
#include "Routine.h"
#include "Util.h"

namespace atomic {


RoutineCreatorTable& GetRoutineCreatorTable()
{
    static RoutineCreatorTable s_routine_creators;
    return s_routine_creators;
}


IRoutine* CreateRoutine(RoutineClassID rcid)
{
    if(rcid==RCID_Null) { return NULL; }
    return GetRoutineCreatorTable()[rcid]();
}


class dpPatch Routine_SingleShoot : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    int32 m_frame;
    int32 m_cycle;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_frame)
        istSerialize(m_cycle)
        )

public:
    atomicECallBlock(
        atomicECallSuper(mhandler)
        )

public:
    Routine_SingleShoot() : m_frame(0), m_cycle(0) {}

    void update(float32 dt)
    {
        ++m_frame;
        IEntity *e = getEntity();
        vec4 pos; atomicQuery(e, getPosition, pos);
        vec4 player_pos = GetNearestPlayerPosition(pos);
        vec4 dir = vec4(glm::normalize(vec2(player_pos)-vec2(pos)), 0.0f, 0.0f);
        if(m_frame % 120 == 0) {
            for(int i=0; i<3; ++i) {
                vec4 vel = (dir*(0.008f + 0.002f*i));
                ShootSimpleBullet(e->getHandle(), pos, vel);
            }
            ++m_cycle;
       }
        pos += dir*0.0035f;
        atomicCall(e, setPosition, pos);
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        vec4 v = m->direction * m->direction.w * 0.02f;
        IEntity *e = getEntity();
        vec4 pos; atomicQuery(e, getPosition, pos);
        pos += v;
        pos.z = 0.0f;
        pos.w = 0.0f;
        atomicCall(e, setPosition, pos);
    }
};
atomicImplementRoutine(Routine_SingleShoot);
atomicExportClass(atomic::Routine_SingleShoot);

class dpPatch Routine_CircularShoot : public IRoutine
{
typedef IRoutine super;
private:
    int32 m_frame;
    int32 m_cycle;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_frame)
        istSerialize(m_cycle)
        )

public:
    Routine_CircularShoot() : m_frame(0), m_cycle(0) {}

    void update(float32 dt)
    {
        ++m_frame;
        IEntity *e = getEntity();
        vec4 pos; atomicQuery(e, getPosition, pos);
        vec4 dir = vec4(0.0f, 1.0f, 0.0f, 0.0f);
        if(m_frame % 20 == 0) {
            for(int i=0; i<10; ++i) {
                mat4 rot = glm::rotate(mat4(), -90.0f+10.0f*m_cycle, vec3(0.0f,0.0f,1.0f));
                vec4 vel = rot * (dir*(0.008f + 0.001f*i));
                ShootSimpleBullet(e->getHandle(), pos, vel);
            }
            ++m_cycle;
        }
    }
};
atomicImplementRoutine(Routine_CircularShoot);
atomicExportClass(atomic::Routine_CircularShoot);


class dpPatch Routine_HomingPlayer : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    vec4 m_vel;
    vec4 m_target_pos;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_vel)
        istSerialize(m_target_pos)
        )

public:
    atomicECallBlock(
        atomicECallSuper(mhandler)
    )

public:
    Routine_HomingPlayer() {}

    void update(float32 dt)
    {
        IEntity *e = getEntity();
        vec4 pos; atomicQuery(e, getPosition, pos);
        m_target_pos = GetNearestPlayerPosition(pos);
    }

    void asyncupdate(float32 dt)
    {
        IEntity *e = getEntity();
        vec4 pos; atomicQuery(e, getPosition, pos);
        m_vel *= 0.98f;
        m_vel += glm::normalize(m_target_pos-pos) * 0.0002f;
        pos += m_vel;
        atomicCall(e, setPosition, pos);
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        vec4 v = m->direction * m->direction.w * 0.1f;
        m_vel += v;
        m_vel.z = 0.0f;
        m_vel.w = 0.0f;

        float32 len = glm::length(m_vel);
        const float32 max_speed = 0.01f;
        if(len > max_speed) { m_vel = m_vel / len * max_speed; }
    }
};
atomicImplementRoutine(Routine_HomingPlayer);
atomicExportClass(atomic::Routine_HomingPlayer);


class dpPatch Routine_Pinball : public IRoutine, public Attr_MessageHandler
{
typedef Routine_Pinball this_t;
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    vec4 m_vel;
    vec4 m_accel;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_vel)
        istSerialize(m_accel)
        )

public:
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setVelocity)
        atomicECall(setAccel)
        )
        atomicECallSuper(mhandler)
    )

public:
    Routine_Pinball() {}

    void setVelocity(const vec4 &v) { m_vel=v; }
    void setAccel(const vec4 &v)    { m_accel=v; }

    void asyncupdate(float32 dt)
    {
        IEntity *e = getEntity();
        vec4 pos; atomicQuery(e, getPosition, pos);
        pos += m_vel;
        m_vel += m_accel;
        atomicCall(e, setPosition, pos);
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        vec4 v = m->direction * m->direction.w * 0.2f;
        m_vel += v;
        m_vel.z = 0.0f;
        m_vel.w = 0.0f;

        float32 len = glm::length(m_vel);
        const float32 max_speed = 0.01f;
        if(len > max_speed) { m_vel = m_vel / len * max_speed; }

        //vec4 base_dir = m->direction;
        //vec4 dir = base_dir;
        //dir.z = dir.w = 0.0f;
        //dir = glm::normalize(dir);

        //vec4 vel = m_vel;
        //if(glm::dot(dir, vel) < 0.0f) {
        //    m_vel = glm::reflect(vel*0.98f, dir);
        //}
    }
};
atomicImplementRoutine(Routine_Pinball);
atomicExportClass(atomic::Routine_Pinball);

} // namespace atomic
