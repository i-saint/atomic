#include "stdafx.h"
#include "types.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/CollisionModule.h"
#include "Game/EntityModule.h"
#include "Game/EntityQuery.h"
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
public:
    enum State {
        State_Normal,
        State_Fadeout,
        State_Dead,
    };
private:
    typedef stl::vector<LaserParticle> particles;

    LaserHandle     m_handle;
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
    Laser(LaserHandle handle, const vec3 &pos, const vec3 &dir, EntityHandle owner)
        : m_handle(handle), m_owner(0), m_group(0), m_state(State_Normal), m_time(0.0f)
    {
        m_pos = pos;
        m_dir = dir;
        m_owner = owner;
        atmQuery(owner, getCollisionGroup, m_group);
    }

    State getState() const { return m_state; }
    LaserHandle getHandle() const override      { return m_handle; }
    const vec3& getPosition() const override    { return m_pos; }
    const vec3& getDirection() const override   { return m_dir; }
    void setPosition(const vec3 &v) override    { m_pos=v; }
    void setDirection(const vec3 &v) override   { m_dir=v; }
    void fade() override { m_state=State_Fadeout; m_time=0.0f; }
    void kill() override { m_state=State_Dead; m_time=0.0f; }
    bool isDead() const { return m_state==State_Dead; }

    void update(float32 dt)
    {
        m_time += dt;

        static const float32 speed = 0.1f;
        static const float32 lifetime = 240.0f;
        static const float32 radius = 0.02f;
        each(m_particles, [&](LaserParticle &p){
            p.time += dt;
        });

        if(m_state==State_Normal) {
            CollisionSphere sphere;
            sphere.setEntityHandle(m_owner);
            sphere.setCollisionGroup(m_group);
            each(m_particles, [&](LaserParticle &p){
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
        erase(m_particles, [&](LaserParticle &p){ return p.time>=lifetime; });
    }

    void asyncupdate(float32 dt)
    {
    }

    void draw()
    {
        // todo
    }

};

class dpPatch LaserManager : public IBulletManager
{
private:
    typedef stl::vector<Laser*> Lasers;
    typedef stl::vector<LaserHandle> Handles;

    Lasers m_lasers;
    Handles m_all;
    Handles m_vacants;
    Handles m_dead;
    Handles m_dead_prev;

    istSerializeBlock(
        istSerialize(m_lasers)
        istSerialize(m_all)
        istSerialize(m_vacants)
        istSerialize(m_dead)
        istSerialize(m_dead_prev)
    )

public:
    LaserManager()
    {
    }

    ~LaserManager()
    {
        each(m_lasers, [&](Laser *v){ istDelete(v); });
        m_lasers.clear();
    }

    Laser* createLaser(const vec3 &pos, const vec3 &dir, EntityHandle owner)
    {
        atmDbgAssertSyncLock();
        LaserHandle h = 0;
        if(!m_vacants.empty()) {
            h = m_vacants.back();
            m_vacants.pop_back();
        }
        else {
            h = m_lasers.size();
            m_lasers.push_back(nullptr);
        }

        Laser *l = istNew(Laser)(h, pos, dir, owner);
        m_lasers[h] = l;
        m_all.push_back(h);
        return l;
    }

    void update(float32 dt) override
    {
        each(m_all, [&](LaserHandle &h){
            Laser *&v = m_lasers[h];
            if(v) {
                v->update(dt);
                if(v->isDead()) {
                    istSafeDelete(v);
                }
            }
            if(!v) {
                m_dead.push_back(h);
                h = 0;
            }
        });
        erase(m_all, [&](LaserHandle h){ return h==0; });

        m_vacants.insert(m_vacants.end(), m_dead_prev.begin(), m_dead_prev.end());
        m_dead_prev = m_dead;
        m_dead.clear();
    }

    void asyncupdate(float32 dt) override
    {
        each(m_all, [&](LaserHandle h){
            if(Laser *v = m_lasers[h]) {
                v->asyncupdate(dt);
            }
        });
    }

    void draw() override
    {
        each(m_all, [&](LaserHandle h){
            if(Laser *v = m_lasers[h]) {
                v->draw();
            }
        });
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

    BulletData() : time(), owner(), hit_to(), flags() {}
    BulletData(const vec3 &p, const vec3 &v, EntityHandle o) : pos(p), vel(v), owner(o)
    {
        atmQuery(owner, getCollisionGroup, group);
    }
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
        m_bullets.reserve(512);
    }

    void update(float32 dt) override
    {
        static const float32 radius = 0.03f;
        static const float32 lifetime = 600.0f;

        CollisionSphere sphere;
        each(m_bullets, [&](BulletData &p){
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
        erase(m_bullets, [&](BulletData &p){ return p.time>=lifetime; });
    }

    void asyncupdate(float32 dt) override
    {

    }

    void draw() override
    {
        vec4 diffuse = vec4(0.6f, 0.6f, 0.6f, 80.0f);
        vec4 glow = vec4(2.0f, 1.2f, 0.1f, 0.0f);
        vec4 light  = glow;
        vec4 flash  = glow * 0.5f;
        vec3 axis1(0.0f, 1.0f, 0.0f);
        vec3 axis2(0.0f, 0.0f, 1.0f);
        if(atmGetConfig()->lighting<atmE_Lighting_High) {
            flash  = glow * 0.7f;
        }

        PSetInstance inst;
        inst.diffuse = diffuse;
        inst.glow = glow;
        inst.flash = vec4();
        inst.elapsed = 0.0f;
        inst.appear_radius = 10000.0f;
        each(m_bullets, [&](BulletData &p){
            mat4 mat;
            mat = glm::translate(mat, p.pos);
            mat = glm::rotate(mat, 4.5f*p.time, axis1);
            mat = glm::rotate(mat, 4.5f*p.time, axis2);
            inst.translate = mat;
            atmGetSPHPass()->addPSetInstance(PSET_SPHERE_BULLET, inst);
        });
        if(atmGetConfig()->lighting>=atmE_Lighting_High) {
            each(m_bullets, [&](BulletData &p){
                PointLight l;
                l.setPosition(p.pos + vec3(0.0f, 0.0f, 0.10f));
                l.setRadius(0.2f);
                l.setColor(light);
                atmGetLightPass()->addLight(l);
            });
            PointLight l;
        }
    }

    void shoot(const vec3 &pos, const vec3 &vel, EntityHandle owner)
    {
        atmDbgAssertSyncLock();
        BulletData bd(pos, vel, owner);
        m_bullets.push_back(bd);
    }
};



class dpPatch NeedleManager
{
private:
public:
};



BulletModule::BulletModule()
    : m_lasers(), m_bullets()
{
}

BulletModule::~BulletModule()
{
    each(m_managers, [&](IBulletManager *bm){ istDelete(bm); });
}

void BulletModule::initialize()
{
    m_lasers = istNew(LaserManager)();
    m_managers.push_back(m_lasers);

    m_bullets = istNew(BulletManager)();
    m_managers.push_back(m_bullets);
}

void BulletModule::frameBegin()
{
    each(m_managers, [&](IBulletManager *bm){ bm->frameBegin(); });
}

void BulletModule::update(float32 dt)
{
    each(m_managers, [&](IBulletManager *bm){ bm->update(dt); });
}

void BulletModule::asyncupdate(float32 dt)
{
    each(m_managers, [&](IBulletManager *bm){ bm->asyncupdate(dt); });
}

void BulletModule::draw()
{
    each(m_managers, [&](IBulletManager *bm){ bm->draw(); });
}

void BulletModule::frameEnd()
{
    each(m_managers, [&](IBulletManager *bm){ bm->frameEnd(); });
}

void BulletModule::shootBullet(const vec3 &pos, const vec3 &vel, EntityHandle owner)
{
    m_bullets->shoot(pos, vel, owner);
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
