#include "stdafx.h"
#include "types.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/CollisionModule.h"
#include "Game/FluidModule.h"
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
    static const float32 s_speed;
    static const float32 s_lifetime;
    static const float32 s_radius;

    LaserHandle     m_handle;
    EntityHandle    m_owner;
    CollisionGroup  m_group;
    State           m_state;
    float32         m_time;
    vec3            m_pos;
    vec3            m_dir;
    particles       m_particles;

    // 以下 serialize 不要
    CollisionModule::CollisionContext m_cctx;
    ist::vector<SingleParticle> m_drawdata;

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

        wdmScope(
        wdmString path = wdmFormat("Bullet/Laser/0x%p", this);
        wdmAddNode(path+"/m_pos", &m_pos );
        wdmAddNode(path+"/m_dir", this, &Laser::getDirection, &Laser::setDirection );
        wdmAddNode(path+"/fade()", &Laser::fade, this );
        wdmAddNode(path+"/kill()", &Laser::kill, this );
        )
    }

    ~Laser()
    {
        wdmEraseNode(wdmFormat("Bullet/Laser/0x%p", this));
    }

    State getState() const { return m_state; }
    LaserHandle getHandle() const override      { return m_handle; }
    const vec3& getPosition() const override    { return m_pos; }
    const vec3& getDirection() const override   { return m_dir; }
    void setPosition(const vec3 &v) override    { m_pos=v; }
    void setDirection(const vec3 &v) override   { m_dir=glm::normalize(v); }
    void fade() override { m_state=State_Fadeout; m_time=0.0f; }
    void kill() override { m_state=State_Dead; m_time=0.0f; }
    bool isDead() const { return m_state==State_Dead; }

    void update(float32 dt)
    {
        m_time += dt;

        for(size_t i=0; i<3; ++i) {
            LaserParticle t;
            t.pos = m_pos + (vec3(GenRandomVector3()*0.075f)*vec3(1.0f,1.0f,0.5f));
            t.time = 0.0f;
            m_particles.push_back(t);
        }

        const float32 lifetime = 240.0f;
        const float32 power = 2.0f;

        each(m_particles, [&](LaserParticle &p){
            p.time += dt;
        });

        if(m_state==State_Normal) {
            each(m_particles, [&](LaserParticle &p){
                vec3 pos = p.pos + m_dir*p.time*s_speed;
                CollisionSphere sphere;
                sphere.setEntityHandle(m_owner);
                sphere.setCollisionGroup(m_group);
                sphere.pos_r = vec4(pos, s_radius);
                sphere.updateBoundingBox();
                if(atmGetCollisionModule()->collideRecv(&sphere, m_cctx)) {
                    p.hit_to = m_cctx.messages.front().from;
                    p.time = lifetime;
                }
                m_cctx.clear();
            });
        }
        each(m_particles, [&](LaserParticle &p){
            if(p.hit_to) {
                vec3 pos = p.pos + m_dir*p.time*s_speed;
                //atmGetFluidModule()->addFluid(PSET_SPHERE_BULLET, glm::translate(pos));
                atmCall(p.hit_to, damage, power);
            }
        });
        erase(m_particles, [&](LaserParticle &p){ return p.time>=lifetime; });
    }

    void asyncupdate(float32 dt)
    {
    }

    void draw()
    {
        if(m_particles.empty()) { return; }

        //const vec4 color(0.5f,0.5f,0.7f,1.0f);
        //const vec4 glow(0.5f,0.5f,0.7f,1.0f);

        //m_drawdata.resize(m_particles.size());
        //each_with_index(m_particles, [&](LaserParticle &p, size_t i){
        //    SingleParticle &dd = m_drawdata[i];
        //    dd.position = vec4(p.pos+m_dir*p.time*s_speed, 1.0f);
        //    dd.color = color;
        //    dd.glow = glow;
        //    dd.scale = 1.0f;
        //});
        //atmGetParticlePass()->addParticle(&m_drawdata[0], m_drawdata.size());

        vec4 diffuse = vec4(0.6f, 0.6f, 0.6f, 80.0f);
        vec4 glow = vec4(2.0f, 1.2f, 0.1f, 0.0f);
        vec4 light  = glow;
        vec4 flash  = glow * 0.5f;
        if(atmGetConfig()->lighting<atmE_Lighting_High) {
            flash  = glow * 0.7f;
        }

        PSetInstance inst;
        inst.diffuse = diffuse;
        inst.glow = glow;
        inst.flash = flash;
        inst.elapsed = 0.0f;
        inst.appear_radius = 10000.0f;
        each(m_particles, [&](LaserParticle &p){
            vec3 pos = p.pos + m_dir*p.time*s_speed;
            inst.translate = glm::translate(pos);
            atmGetFluidPass()->addPSetInstance(PSET_SPHERE_BULLET, inst);
        });
    }

};
const float32 Laser::s_speed = 0.025f;
const float32 Laser::s_lifetime = 180.0f;
const float32 Laser::s_radius = 0.02f;

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
        m_lasers.push_back(nullptr);
    }

    ~LaserManager()
    {
        each(m_lasers, [&](Laser *v){ istDelete(v); });
        m_lasers.clear();
    }

    LaserHandle createLaser(const vec3 &pos, const vec3 &dir, EntityHandle owner)
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
        return h;
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
    BulletData(const vec3 &p, const vec3 &v, EntityHandle o) : pos(p), vel(v), owner(o), hit_to(), flags()
    {
        atmQuery(owner, getCollisionGroup, group);
    }

    mat4 computeMatrix() const
    {
        const vec3 axis1(0.0f, 1.0f, 0.0f);
        const vec3 axis2(0.0f, 0.0f, 1.0f);
        mat4 mat;
        mat = glm::translate(mat, pos);
        mat = glm::rotate(mat, 4.5f*time, axis1);
        mat = glm::rotate(mat, 4.5f*time, axis2);
        return mat;
    }
};
atmSerializeRaw(BulletData);

class dpPatch BulletManager : public IBulletManager
{
private:
    typedef stl::vector<BulletData> Bullets;
    typedef CollisionModule::CollisionContext CollisionContext;
    typedef stl::vector<CollisionContext> CollisionContexts;
    Bullets m_bullets;
    CollisionContexts m_cctx; // serialize 不要

public:
    BulletManager()
    {
        m_bullets.reserve(512);
    }

    void update(float32 dt) override
    {
        const float32 radius = 0.03f;
        const float32 lifetime = 600.0f;
        const float32 power = 5.0f;
        const size_t blocksize = 32;

        size_t num_tasks = ceildiv(m_bullets.size(), blocksize);
        if(num_tasks>m_cctx.size()) {
            m_cctx.resize(num_tasks);
        }

        parallel_each_with_block_index(m_bullets, blocksize, [&](BulletData &p, size_t bi){
            vec3 pos = p.pos + p.vel;
            pos.z = 0.03f;
            p.time += dt;
            p.pos = pos;

            CollisionContext &ctx = m_cctx[bi];
            CollisionSphere sphere;
            sphere.setEntityHandle(p.owner);
            sphere.setCollisionGroup(p.group);
            sphere.pos_r = vec4(pos, radius);
            sphere.bb.bl = vec4(pos-radius, 0.0f);
            sphere.bb.ur = vec4(pos+radius, 0.0f);
            if(atmGetCollisionModule()->collideRecv(&sphere, ctx)) {
                p.hit_to = ctx.messages.front().from;
                p.time = lifetime;
            }
            ctx.clear();
        });
        each(m_bullets, [&](BulletData &p){
            if(p.hit_to) {
                atmGetFluidModule()->addFluid(PSET_SPHERE_BULLET, p.computeMatrix());
                atmCall(p.hit_to, damage, power);
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
        if(atmGetConfig()->lighting<atmE_Lighting_High) {
            flash  = glow * 0.7f;
        }

        PSetInstance inst;
        inst.diffuse = diffuse;
        inst.glow = glow;
        inst.flash = flash;
        inst.elapsed = 0.0f;
        inst.appear_radius = 10000.0f;
        each(m_bullets, [&](BulletData &p){
            inst.translate = p.computeMatrix();
            atmGetFluidPass()->addPSetInstance(PSET_SPHERE_BULLET, inst);
        });
        if(atmGetConfig()->lighting>=atmE_Lighting_High) {
            each(m_bullets, [&](BulletData &p){
                PointLight l;
                l.setPosition(p.pos + vec3(0.0f, 0.0f, 0.10f));
                l.setRadius(0.2f);
                l.setColor(light);
                atmGetLightPass()->addLight(l);
            });
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
    return m_lasers->createLaser(pos, dir, owner);
}

ILaser* BulletModule::getLaser(LaserHandle v)
{
    return nullptr;
}


} // namespace atm
