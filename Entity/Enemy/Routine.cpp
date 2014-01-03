#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Routine.h"

namespace atm {


RoutineCreatorTable& GetRoutineCreatorTable()
{
    static RoutineCreatorTable s_routine_creators;
    return s_routine_creators;
}


IRoutine* CreateRoutine(RoutineClassID rcid)
{
    if(rcid==RCID_Null) { return nullptr; }
    return GetRoutineCreatorTable()[rcid]();
}


class Routine_SingleShoot : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    enum State {
        State_Normal,
        State_Laser,
    };
    float32 m_time;
    int32 m_cycle;
    State m_state;
    LaserHandle m_laser;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_time)
        istSerialize(m_cycle)
        istSerialize(m_state)
        istSerialize(m_laser)
    )

public:
    atmECallBlock(
        atmECallSuper(mhandler)
        atmMethodBlock(
            atmECall(instruct)
        )
    )

public:
    Routine_SingleShoot() : m_time(0.0f), m_cycle(0), m_state(State_Normal), m_laser(0)
    {}

    void setState(State v) { m_state=v; m_time=0.0f; }
    State getState() const { return m_state; }

    void finalize() override
    {
        if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
            l->kill();
        }
    }

    void update(float32 dt) override
    {
        switch(getState()) {
        case State_Normal: update_Normal(dt); break;
        case State_Laser: update_Laser(dt); break;
        }
    }

    void update_Normal(float32 dt)
    {
        m_time += dt;
        IEntity *e = getEntity();
        vec3 pos; atmQuery(e, getPosition, pos);
        vec3 player_pos = GetNearestPlayerPosition(pos);
        vec3 dir = vec3(glm::normalize(vec2(player_pos)-vec2(pos)), 0.0f);
        if(moddiv(m_time, 150.0f)) {
            for(int i=0; i<5; ++i) {
                vec3 vel = (dir*(0.007f + 0.0015f*i));
                ShootSimpleBullet(e->getHandle(), pos, vel);
            }
            ++m_cycle;
        }
        pos += dir*0.0035f*dt;
        atmCall(e, setPosition, pos);
    }

    void update_Laser(float32 dt)
    {
        m_time += dt;
        if(m_time>100.0f) {
            if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
                l->fade();
            }
            else {
                m_laser = 0;
                setState(State_Normal);
            }
        }
    }

    void instruct(const vec3 &tpos, EntityHandle tobj)
    {
        if(getState()!=State_Laser) {
            setState(State_Laser);

            IEntity *e = getEntity();
            vec3 pos; atmQuery(e, getPosition, pos);
            vec3 dir = glm::normalize(tpos-pos);
            m_laser = atmGetBulletModule()->createLaser(pos+dir*0.2f, dir, e->getHandle());
        }
    }

    void eventCollide(const CollideMessage *m) override
    {
        IEntity *e = getEntity();
        vec3 v = glm::normalize(vec3(m->direction.x,m->direction.y,0.0f)) * (m->direction.w * 0.1f);
        vec3 pos; atmQuery(e, getPosition, pos);
        pos += v;
        pos.z = 0.0f;
        atmCall(e, setPosition, pos);
    }
};
atmImplementRoutine(Routine_SingleShoot);
atmExportClass(Routine_SingleShoot);


class Routine_FixedShotgun : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    enum State {
        State_Normal,
        State_Laser,
    };
    float32 m_time;
    int32 m_cycle;
    State m_state;
    LaserHandle m_laser;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_time)
        istSerialize(m_cycle)
        istSerialize(m_state)
        istSerialize(m_laser)
    )

public:
    atmECallBlock(
        atmECallSuper(mhandler)
        atmMethodBlock(
            atmECall(instruct)
        )
    )

public:
    Routine_FixedShotgun() : m_time(0.0f), m_cycle(0), m_state(State_Normal), m_laser(0)
    {}

    void setState(State v) { m_state=v; m_time=0.0f; }
    State getState() const { return m_state; }

    void finalize() override
    {
        if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
            l->kill();
        }
    }

    void update(float32 dt) override
    {
        switch(getState()) {
        case State_Normal: update_Normal(dt); break;
        case State_Laser: update_Laser(dt); break;
        }
    }

    void update_Normal(float32 dt)
    {
        m_time += dt;
        IEntity *e = getEntity();
        if(moddiv(m_time, 300.0f)) {
            vec3 pos = atmGetProperty(vec3, e, getPositionAbs);
            vec3 player_pos = GetNearestPlayerPosition(pos);
            float32 d = glm::length(vec2(player_pos)-vec2(pos));
            if(d < 4.0f) {
                vec3 dir = vec3(glm::normalize(vec2(player_pos)-vec2(pos)), 0.0f);
                for(int i=0; i<20; ++i) {
                    vec3 vel = (dir+GenRandomUnitVector3()*0.2f)*0.012f;
                    vel.z = 0.0f;
                    ShootSimpleBullet(e->getHandle(), pos, vel);
                }
            }
            ++m_cycle;
        }
    }

    void update_Laser(float32 dt)
    {
        m_time += dt;
        if(m_time>100.0f) {
            if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
                l->fade();
            }
            else {
                m_laser = 0;
                setState(State_Normal);
            }
        }
    }

    void instruct(const vec3 &tpos, EntityHandle tobj)
    {
        if(getState()!=State_Laser) {
            setState(State_Laser);

            IEntity *e = getEntity();
            vec3 pos = atmGetProperty(vec3, e, getPositionAbs);
            vec3 dir = glm::normalize(tpos-pos);
            m_laser = atmGetBulletModule()->createLaser(pos+dir*0.2f, dir, e->getHandle());
        }
    }
};
atmImplementRoutine(Routine_FixedShotgun);
atmExportClass(Routine_FixedShotgun);



class Routine_FixedShotgun2 : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    enum State {
        State_Normal,
        State_Laser,
    };
    float32 m_time;
    int32 m_cycle;
    State m_state;
    LaserHandle m_laser;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_time)
        istSerialize(m_cycle)
        istSerialize(m_state)
        istSerialize(m_laser)
    )

public:
    atmECallBlock(
        atmECallSuper(mhandler)
        atmMethodBlock(
            atmECall(instruct)
        )
    )

public:
    Routine_FixedShotgun2() : m_time(0.0f), m_cycle(0), m_state(State_Normal), m_laser(0)
    {}

    void setState(State v) { m_state=v; m_time=0.0f; }
    State getState() const { return m_state; }

    void finalize() override
    {
        if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
            l->kill();
        }
    }

    void update(float32 dt) override
    {
        switch(getState()) {
        case State_Normal: update_Normal(dt); break;
        case State_Laser: update_Laser(dt); break;
        }
    }

    void update_Normal(float32 dt)
    {
        m_time += dt;
        IEntity *e = getEntity();
        if(moddiv(m_time, 200.0f)) {
            vec3 pos = atmGetProperty(vec3, e, getPositionAbs);
            vec3 player_pos = GetNearestPlayerPosition(pos);
            float32 d = glm::length(vec2(player_pos)-vec2(pos));
            if(d < 4.0f) {
                vec3 dir = vec3(glm::normalize(vec2(player_pos)-vec2(pos)), 0.0f);
                for(int i=0; i<10; ++i) {
                    vec3 vel = (dir+GenRandomUnitVector3()*0.2f)*0.012f;
                    vel.z = 0.0f;
                    ShootSimpleBullet(e->getHandle(), pos, vel);
                }
            }
            ++m_cycle;
        }
    }

    void update_Laser(float32 dt)
    {
        m_time += dt;
        if(m_time>100.0f) {
            if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
                l->fade();
            }
            else {
                m_laser = 0;
                setState(State_Normal);
            }
        }
    }

    void instruct(const vec3 &tpos, EntityHandle tobj)
    {
        if(getState()!=State_Laser) {
            setState(State_Laser);

            IEntity *e = getEntity();
            vec3 pos = atmGetProperty(vec3, e, getPositionAbs);
            vec3 dir = glm::normalize(tpos-pos);
            m_laser = atmGetBulletModule()->createLaser(pos+dir*0.2f, dir, e->getHandle());
        }
    }
};
atmImplementRoutine(Routine_FixedShotgun2);
atmExportClass(Routine_FixedShotgun2);


class Routine_FixedLaser : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    enum State {
        State_Normal,
        State_Laser,
    };
    float32 m_time;
    int32 m_cycle;
    State m_state;
    LaserHandle m_laser;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_time)
        istSerialize(m_cycle)
        istSerialize(m_state)
        istSerialize(m_laser)
    )

public:
    atmECallBlock(
        atmECallSuper(mhandler)
            atmMethodBlock(
            atmECall(instruct)
        )
    )

public:
    Routine_FixedLaser() : m_time(0.0f), m_cycle(0), m_state(State_Normal), m_laser(0)
    {}

    void setState(State v) { m_state=v; m_time=0.0f; }
    State getState() const { return m_state; }

    void finalize() override
    {
        if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
            l->kill();
        }
    }

    void update(float32 dt) override
    {
        switch(getState()) {
        case State_Normal: update_Normal(dt); break;
        case State_Laser: update_Laser(dt); break;
        }
    }

    void update_Normal(float32 dt)
    {
        m_time += dt;
        IEntity *e = getEntity();
        if(moddiv(m_time, 400.0f)) {
            setState(State_Laser);

            IEntity *e = getEntity();
            vec3 pos = atmGetProperty(vec3, e, getPositionAbs);
            vec3 tpos = GetNearestPlayerPosition(pos);
            vec3 dir = glm::normalize(tpos-pos);
            m_laser = atmGetBulletModule()->createLaser(pos+dir*0.2f, dir, e->getHandle());
        }
    }

    void update_Laser(float32 dt)
    {
        m_time += dt;
        if(m_time>100.0f) {
            if(ILaser *l=atmGetBulletModule()->getLaser(m_laser)) {
                l->fade();
            }
            else {
                m_laser = 0;
                setState(State_Normal);
            }
        }
    }

    void instruct(const vec3 &tpos, EntityHandle tobj)
    {
        if(getState()!=State_Laser) {
            setState(State_Laser);

            IEntity *e = getEntity();
            vec3 pos = atmGetProperty(vec3, e, getPositionAbs);
            vec3 dir = glm::normalize(tpos-pos);
            m_laser = atmGetBulletModule()->createLaser(pos+dir*0.2f, dir, e->getHandle());
        }
    }
};
atmImplementRoutine(Routine_FixedLaser);
atmExportClass(Routine_FixedLaser);


class Routine_CircularShoot : public IRoutine
{
typedef IRoutine super;
private:
    float32 m_time;
    int32 m_cycle;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_time)
        istSerialize(m_cycle)
    )

public:
    Routine_CircularShoot() : m_time(0.0f), m_cycle(0) {}

    void update(float32 dt)
    {
        m_time += dt;
        IEntity *e = getEntity();
        vec3 pos; atmQuery(e, getPosition, pos);
        vec4 dir = vec4(0.0f, 1.0f, 0.0f, 1.0);
        if(moddiv(m_time, 20.0f)) {
            for(int i=0; i<10; ++i) {
                mat4 rot = glm::rotate(mat4(), -90.0f+10.0f*m_cycle, vec3(0.0f,0.0f,1.0f));
                vec3 vel = vec3( rot * (dir*(0.008f + 0.001f*i)) );
                ShootSimpleBullet(e->getHandle(), pos, vel);
            }
            ++m_cycle;
        }
    }
};
atmImplementRoutine(Routine_CircularShoot);
atmExportClass(Routine_CircularShoot);


class Routine_HomingPlayer : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    float32 m_time;
    vec3 m_vel;
    vec3 m_target_pos;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_time)
        istSerialize(m_vel)
        istSerialize(m_target_pos)
    )

public:
    atmECallBlock(
        atmECallSuper(mhandler)
        atmMethodBlock(
            atmECall(instruct)
        )
    )

public:
    Routine_HomingPlayer() : m_time(0.0f)
    {}

    void update(float32 dt)
    {
        m_time += dt;

        IEntity *e = getEntity();
        vec3 pos; atmQuery(e, getPosition, pos);
        m_target_pos = GetNearestPlayerPosition(pos);

        //if(moddiv(m_time, 10.0f)) {
        //    vec3 vel = glm::normalize(m_target_pos-pos)*0.015f;
        //    ShootSimpleBullet(e->getHandle(), pos, vel);
        //}
    }

    void asyncupdate(float32 dt)
    {
        IEntity *e = getEntity();
        vec3 pos; atmQuery(e, getPosition, pos);
        m_vel *= 0.98f;
        m_vel += glm::normalize(m_target_pos-pos) * 0.0002f;
        pos += m_vel*dt;
        atmCall(e, setPosition, pos);
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        vec3 v = vec3(m->direction * m->direction.w * 0.1f);
        m_vel += v;
        m_vel.z = 0.0f;

        float32 len = glm::length(m_vel);
        const float32 max_speed = 0.01f;
        if(len > max_speed) { m_vel = m_vel / len * max_speed; }
    }

    void instruct(const vec3 &tpos, EntityHandle tobj)
    {
        IEntity *e = getEntity();
        vec3 pos; atmQuery(e, getPosition, pos);
        vec3 vel = glm::normalize(tpos-pos)*0.015f;
        ShootSimpleBullet(e->getHandle(), pos, vel);
    }
};
atmImplementRoutine(Routine_HomingPlayer);
atmExportClass(Routine_HomingPlayer);


class Routine_Pinball : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    vec3 m_vel;
    vec3 m_accel;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_vel)
        istSerialize(m_accel)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(setVelocity)
            atmECall(setAccel)
        )
        atmECallSuper(mhandler)
    )

public:
    Routine_Pinball() {}

    void setVelocity(const vec3 &v) { m_vel=v; }
    void setAccel(const vec3 &v)    { m_accel=v; }

    void asyncupdate(float32 dt)
    {
        IEntity *e = getEntity();
        vec3 pos; atmQuery(e, getPosition, pos);
        pos += m_vel * dt;
        m_vel += m_accel;
        atmCall(e, setPosition, pos);
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        vec3 v = vec3(m->direction * m->direction.w * 0.2f);
        m_vel += v;
        m_vel.z = 0.0f;

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
atmImplementRoutine(Routine_Pinball);
atmExportClass(Routine_Pinball);






inline IEntity* PutSmallEnemy()
{
    IEntity *e = nullptr;
    e = atmCreateEntityT(Enemy_Test);
    atmCall(e, setCollisionShape, CS_Sphere);
    atmCall(e, setModel, PSET_SPHERE_SMALL);
    atmCall(e, setPosition, vec3(GenRandomVector2()*2.2f, 0.0f));
    atmCall(e, setLife, 25.0f);
    atmCall(e, setAxis1, GenRandomUnitVector3());
    atmCall(e, setAxis2, GenRandomUnitVector3());
    atmCall(e, setRotateSpeed1, 2.4f);
    atmCall(e, setRotateSpeed2, 2.4f);
    atmCall(e, setRoutine, RCID_Routine_HomingPlayer);
    atmCall(e, setLightRadius, 0.5f);
    return e;
}

class Routine_AlcantareaDemo : public IRoutine, public Attr_MessageHandler
{
typedef IRoutine super;
typedef Attr_MessageHandler mhandler;
private:
    int32 m_frame;
    vec3 m_vel;
    vec3 m_target_pos;
    int32 m_cycle;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerialize(m_frame)
        istSerialize(m_vel)
        istSerialize(m_target_pos)
        istSerialize(m_cycle)
    )

public:
    atmECallBlock(
        atmECallSuper(mhandler)
    )

public:
    Routine_AlcantareaDemo() : m_frame(), m_cycle()
    {}

    void update(float32 dt)
    {
        m_frame++;

        IEntity *e = getEntity();
        vec3 pos; atmQuery(e, getPosition, pos);
        m_target_pos = GetNearestPlayerPosition(pos);

        if(m_frame%5==0) {
            ++m_cycle;
            for(int i=0; i<10; ++i) {
                vec3 vel = glm::normalize(m_target_pos-pos) * (0.01f+0.001f*i);
                vel = glm::rotate(vel, 10.0f*m_cycle, vec3(0.0f,0.0f,1.0f));
                ShootSimpleBullet(e->getHandle(), pos, vel);
            }
        }
        //if(m_frame%25==0) {
        //    vec3 dir = glm::normalize(m_target_pos-pos);
        //    vec3 pos = GenRandomUnitVector3()*0.8f + pos;
        //    pos.z = 0.0f;
        //    IEntity *t = atmCreateEntityT(InvisibleLaserTurret);
        //    atmCall(t, setPosition, pos);
        //    atmCall(t, setDirection, dir);
        //    atmCall(t, setOwner, e->getHandle());
        //}
        //if(m_frame%30==0) {
        //    PutSmallEnemy();
        //}
    }
};
atmImplementRoutine(Routine_AlcantareaDemo);
atmExportClass(Routine_AlcantareaDemo);

} // namespace atm
