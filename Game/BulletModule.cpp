#include "stdafx.h"
#include "types.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/CollisionModule.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Util.h"
#include "BulletModule.h"

namespace atm {


struct LaserParticle
{
    vec3    pos;
    float32 time;
    EntityHandle hit_to;

    LaserParticle() : time(0.0f), hit_to(0) {}
};
atmSerializeRaw(LaserParticle);

class dpPatch Laser : public ILaser
{
private:
    enum State {
        State_Normal,
        State_Fadeout,
        State_Dead,
    };
    typedef stl::vector<LaserParticle> particles;

    uint32          m_id;
    EntityHandle    m_owner;
    CollisionGroup  m_group;
    State           m_state;
    float32         m_time;
    vec3            m_pos;
    vec3            m_dir;
    particles       m_particles;
    CollisionModule::CollisionContext m_cctx; // serialize 不要

    istSerializeBlock(
        istSerialize(m_id)
        istSerialize(m_owner)
        istSerialize(m_group)
        istSerialize(m_state)
        istSerialize(m_time)
        istSerialize(m_pos)
        istSerialize(m_dir)
        istSerialize(m_particles)
    )

public:
    Laser() : m_id(0), m_owner(0), m_group(0), m_state(State_Normal), m_time(0.0f)
    {
    }

    virtual const vec3& getPosition() const     { return m_pos; }
    virtual const vec3& getDirection() const    { return m_dir; }
    virtual void setPosition(const vec3 &v)     { m_pos=v; }
    virtual void setDirection(const vec3 &v)    { m_dir=v; }
    virtual void fade() { m_state=State_Fadeout; m_time=0.0f; }
    virtual void kill() { m_state=State_Dead; m_time=0.0f; }

    void update(float32 dt)
    {
        m_time += dt;

        static const float32 speed = 0.1f;
        static const float32 lifetime = 240.0f;
        static const float32 radius = 0.02f;
        std::for_each(m_particles.begin(), m_particles.end(), [&](LaserParticle &p){
            p.time += dt;
        });

        if(m_state==State_Normal) {
            CollisionSphere sphere;
            sphere.setEntityHandle(m_owner);
            sphere.setCollisionGroup(m_group);
            std::for_each(m_particles.begin(), m_particles.end(), [&](LaserParticle &p){
                vec3 pos = p.pos + m_dir*p.time*speed;
                sphere.pos_r = vec4(pos, radius);
                sphere.bb.bl = vec4(pos-radius, 0.0f);
                sphere.bb.ur = vec4(pos+radius, 0.0f);
                if(atmGetCollisionModule()->collide(&sphere, m_cctx)) {
                    p.hit_to = m_cctx.messages.front().to;
                    p.time = lifetime;
                }
            });
        }
        m_particles.erase(
            std::remove_if(m_particles.begin(), m_particles.end(), [&](LaserParticle &p){ return p.time>=lifetime; }),
            m_particles.end());
    }

    void draw()
    {
        // todo
    }

};

class dpPatch LaserManager : public IBulletManager
{
private:
    typedef stl::vector<Laser*> lasers;
    lasers m_lasers;

public:
    LaserManager()
    {
    }

    ~LaserManager()
    {
        each(m_lasers, [&](Laser *v){ istDelete(v); });
        m_lasers.clear();
    }

    void update(float32 dt) override
    {
    }

    void asyncupdate(float32 dt) override
    {
    }

    void draw() override
    {
    }
};



struct BulletData
{
    vec3 pos;
    vec3 vel;
    float32 time;
    EntityHandle owner;
    CollisionGroup group;
    EntityHandle hit_to;
    uint32 flags;
};
atmSerializeRaw(BulletData);

class dpPatch BulletManager : public IBulletManager
{
private:
    typedef stl::vector<BulletData> bullets;
    bullets m_bullets;
    CollisionModule::CollisionContext m_cctx; // serialize 不要

public:
    BulletManager()
    {
    }

    void update(float32 dt) override
    {
        static const float32 radius = 0.03f;
        static const float32 lifetime = 600.0f;

        CollisionSphere sphere;
        std::for_each(m_bullets.begin(), m_bullets.end(), [&](BulletData &p){
            vec3 pos = p.pos + p.vel;
            p.time += dt;
            p.pos = pos;

            sphere.setEntityHandle(p.owner);
            sphere.setCollisionGroup(p.group);
            sphere.pos_r = vec4(pos, radius);
            sphere.bb.bl = vec4(pos-radius, 0.0f);
            sphere.bb.ur = vec4(pos+radius, 0.0f);
            if(atmGetCollisionModule()->collide(&sphere, m_cctx)) {
                p.hit_to = m_cctx.messages.front().to;
                p.time = lifetime;
            }
        });
        m_bullets.erase(
            std::remove_if(m_bullets.begin(), m_bullets.end(), [&](BulletData &p){ return p.time>=lifetime; }),
            m_bullets.end());
    }

    void asyncupdate(float32 dt) override
    {

    }

    void draw() override
    {
        // todo
    }

    void shoot(const vec3 &pos, const vec3 &vel, EntityHandle owner);
};



class dpPatch NeedleManager
{
private:
public:
};



BulletModule::BulletModule()
    : m_lasers(), m_bullets()
{
    m_lasers = istNew(LaserManager)();
    m_managers.push_back(m_lasers);

    m_bullets = istNew(BulletManager)();
    m_managers.push_back(m_bullets);
}

BulletModule::~BulletModule()
{
    each(m_managers, [&](IBulletManager *bm){ istDelete(bm); });
}

void BulletModule::initialize()
{

}

void BulletModule::frameBegin()
{

}

void BulletModule::update(float32 dt)
{

}

void BulletModule::asyncupdate(float32 dt)
{

}

void BulletModule::draw()
{

}

void BulletModule::frameEnd()
{

}

void BulletModule::shootBullet(const vec3 &pos, const vec3 &vel, EntityHandle owner)
{

}

LaserHandle BulletModule::createLaser(const vec3 &pos, const vec3 &dir, EntityHandle owner)
{
    return 0;
}

ILaser* BulletModule::getLaser(LaserHandle v)
{
    return nullptr;
}


} // namespace atm
