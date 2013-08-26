#include "atmPCH.h"
#include "types.h"
#include "Engine/Game/AtomicGame.h"
#include "Engine/Game/World.h"
#include "Engine/Game/CollisionModule.h"
#include "Engine/Game/FluidModule.h"
#include "Engine/Game/EntityModule.h"
#include "Engine/Game/EntityQuery.h"
#include "Engine/Graphics/ResourceManager.h"
#include "Engine/Graphics/Renderer.h"
#include "Util.h"
#include "BulletModule.h"

namespace atm {


struct LaserParticle
{
    vec3    pos_base;
    vec3    pos_current;
    float32 time;
    EntityHandle hit_to;

    LaserParticle() : time(0.0f), hit_to(0) {}
};
atmSerializeRaw(LaserParticle);

class Laser : public ILaser
{
typedef ILaser super;
public:
    enum State {
        State_Normal,
        State_Fadeout,
        State_Dead,
    };
private:
    typedef stl::vector<LaserParticle> Particles;
    static const float32 s_speed;
    static const float32 s_lifetime;
    static const float32 s_fadeout_time;
    static const float32 s_radius;
    static const float32 s_power;

    Particles       m_particles;
    LaserHandle     m_handle;
    EntityHandle    m_owner;
    CollisionGroup  m_group;
    State           m_state;
    float32         m_time;
    vec3            m_pos;
    vec3            m_dir;
    float32         m_light_radius;
    uint32          m_hitcount;

    // 以下 serialize 不要
    typedef CollisionModule::CollisionContext CollisionContext;
    typedef stl::vector<CollisionContext> CollisionContexts;
    CollisionContexts m_cctx;
    ist::vector<SingleParticle> m_drawdata;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_particles)
        istSerialize(m_handle)
        istSerialize(m_owner)
        istSerialize(m_group)
        istSerialize(m_state)
        istSerialize(m_time)
        istSerialize(m_pos)
        istSerialize(m_dir)
        istSerialize(m_light_radius)
        istSerialize(m_hitcount)
    )

public:
    Laser()
    {
    }

    Laser(LaserHandle handle, const vec3 &pos, const vec3 &dir, EntityHandle owner)
        : m_handle(handle), m_owner(0), m_group(0), m_state(State_Normal), m_time(0.0f)
        , m_light_radius(0.75f)
        , m_hitcount(0)
    {
        m_pos = pos;
        m_dir = dir;
        m_owner = owner;
        atmQuery(owner, getCollisionGroup, m_group);

        wdmScope(
        wdmString path = wdmFormat("Bullet/Laser/0x%p", this);
        wdmAddNode(path+"/m_pos", &m_pos, -4.0f, 4.0f );
        wdmAddNode(path+"/m_dir", this, &Laser::getDirection, &Laser::setDirection, -2.0f, 2.0f );
        wdmAddNode(path+"/m_light_radius", &m_light_radius, 0.0f, 2.0f);
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

    void fade() override {
        if(m_state!=State_Fadeout) {
            m_state = State_Fadeout;
            m_time = 0.0f;
        }
    }

    void kill() override {
        m_state=State_Dead;
        m_time=0.0f;
    }

    bool isDead() const {
        return m_state==State_Dead;
    }

    vec3 computeParticlePos(const LaserParticle &p)
    {
        vec3 adv = m_dir * (p.time*s_speed);
        vec3 pos = m_pos + adv;
        mat4 mat;
        mat = glm::translate(mat, pos);
        mat = glm::rotate(mat, p.time*15.0f, m_dir);
        mat = glm::translate(mat, p.pos_base);
        return vec3(mat * vec4(vec3(0.0f), 1.0f));
    }

    void update1(float32 dt)
    {
        m_time += dt;
        if(m_owner && !atmGetEntity(m_owner)) {
            m_state = State_Dead;
            return;
        }

        for(size_t i=0; i<6; ++i) {
            LaserParticle t;
            t.pos_base = GenRandomVector3()*0.075f * vec3(1.0f,1.0f,0.5f);
            t.time = 0.0f;
            m_particles.push_back(t);
        }
        each(m_particles, [&](LaserParticle &p){
            p.time += dt;
        });

        if(m_state==State_Fadeout) {
            if(m_time>s_fadeout_time) {
                m_state = State_Dead;
            }
        }
    }

    void update2async(float32 dt)
    {
        const size_t blocksize = 32;
        size_t num_tasks = ceildiv(m_particles.size(), blocksize);
        if(num_tasks>m_cctx.size()) {
            m_cctx.resize(num_tasks);
        }

        parallel_each_with_block_index(m_particles, blocksize, [&](LaserParticle &p, size_t bi){
            CollisionContext &ctx = m_cctx[bi];
            vec3 pos = computeParticlePos(p);
            p.pos_current = pos;
            CollisionSphere sphere;
            sphere.setEntityHandle(m_owner);
            sphere.setCollisionGroup(m_group);
            sphere.pos_r = vec4(pos, s_radius);
            sphere.updateBoundingBox();
            if(atmGetCollisionModule()->collideRecv(&sphere, ctx)) {
                p.hit_to = ctx.messages.front().from;
                p.time = s_lifetime;
            }
            ctx.clear();
        });
    }

    void update3(float32 dt)
    {
        if(m_state==State_Normal) {
            each(m_particles, [&](LaserParticle &p){
                if(p.hit_to) {
                    ++m_hitcount;
                    vec3 pos = p.pos_current;
                    float32 d = s_power;
                    if(atmIsEnemy(p.hit_to)) { d*=0.5f; }
                    atmCall(p.hit_to, damage, d);
                    atmCall(p.hit_to, pulse, atmArgs(pos, m_dir*s_speed*1000.0f));
                    {
                        psym::Particle particles;
                        istAlign(16) vec4 spos = vec4(pos, 1.0f);
                        particles.position = (psym::simdvec4&)spos;
                        particles.velocity = _mm_set1_ps(0.0f);
                        atmGetFluidModule()->addFluid(&particles, 1);
                    }
                }
            });
        }
        erase(m_particles, [&](LaserParticle &p){ return p.time>=s_lifetime; });
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

        vec4 diffuse = vec4(0.0f, 0.0f, 0.0f, 80.0f);
        vec4 glow = vec4(1.2f, 1.2f, 2.5f, 0.0f);
        vec4 light  = glow;
        vec4 flash  = glow * 0.75f;
        if(atmGetConfig()->lighting_level<atmE_Lighting_High) {
            flash  = glow * 1.0f;
        }

        PSetInstance inst;
        inst.diffuse = diffuse;
        inst.glow = glow;
        inst.flash = flash;
        inst.elapsed = 0.0f;
        inst.appear_radius = 10000.0f;
        inst.rotate = glm::orientation(m_dir, vec3(1.0f,0.0f,0.0f));
        inst.scale = 1.5f;
        if(m_state==State_Fadeout) {
            inst.scale *= std::max<float32>(1.0f-m_time/s_fadeout_time, 0.0f);
        }
        each(m_particles, [&](LaserParticle &p){
            vec3 pos = p.pos_current;
            vec3 scale = vec3(2.0f);
            mat4 mat;
            mat = glm::translate(mat, p.pos_current);
            mat = glm::scale(mat, scale);
            inst.transform = mat;
            atmGetFluidPass()->addParticlesSolid(PSET_SPHERE_BULLET, inst);
        });

        if(atmGetConfig()->lighting_level>=atmE_Lighting_Medium) {
            PointLight l;
            l.setPosition(m_pos + vec3(0.0f, 0.0f, 0.3f));
            l.setColor(flash*inst.scale);
            l.setRadius(m_light_radius);
            atmGetLightPass()->addLight(l);
        }
    }

};
const float32 Laser::s_speed = 0.06f;
const float32 Laser::s_lifetime = 100.0f;
const float32 Laser::s_fadeout_time = 50.0f;
const float32 Laser::s_radius = 0.04f;
const float32 Laser::s_power = 1.0f;
atmExportClass(Laser)

class LaserManager : public IBulletManager
{
typedef IBulletManager super;
private:
    typedef stl::vector<Laser*> Lasers;
    typedef stl::vector<LaserHandle> Handles;

    Lasers m_lasers;
    Handles m_all;
    Handles m_vacants;
    Handles m_dead;
    Handles m_dead_prev;

    istSerializeBlock(
        istSerializeBase(super)
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

    Laser* getLaser(EntityHandle h)
    {
        return h<m_lasers.size() ? m_lasers[h] : nullptr;
    }

    void update(float32 dt) override
    {
        each(m_all, [&](LaserHandle &h){
            Laser *&v = m_lasers[h];
            if(v) {
                v->update1(dt);
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
        parallel_each(m_all, 1, [&](LaserHandle h){
            Laser *&v = m_lasers[h];
            if(v) {
                v->update2async(dt);
            }
        });
        each(m_all, [&](LaserHandle h){
            Laser *&v = m_lasers[h];
            if(v) {
                v->update3(dt);
            }
        });

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

    template<class F>
    void eachLasers(const F &f) {
        each(m_all, [&](LaserHandle h){
            Laser *v = m_lasers[h];
            if(v) {
                f(v);
            }
        });
    }
};
atmExportClass(LaserManager)



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

    mat4 computeTransformMatrix() const
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

class BulletManager : public IBulletManager
{
typedef IBulletManager super;
private:
    typedef stl::vector<BulletData> Bullets;
    typedef CollisionModule::CollisionContext CollisionContext;
    typedef stl::vector<CollisionContext> CollisionContexts;
    Bullets m_bullets;
    CollisionContexts m_cctx; // serialize 不要

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_bullets)
    )

public:
    BulletManager()
    {
        m_bullets.reserve(512);
    }

    void update(float32 dt) override
    {
        const size_t blocksize = 32;
        const float32 radius = 0.03f;
        const float32 lifetime = 600.0f;
        const float32 power = 20.0f;

        size_t num_tasks = ceildiv(m_bullets.size(), blocksize);
        if(num_tasks>m_cctx.size()) {
            m_cctx.resize(num_tasks);
        }

        parallel_each_with_block_index(m_bullets, blocksize, [&](BulletData &p, size_t bi){
            vec3 pos = p.pos + p.vel*dt;
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
                float32 d = power;
                if(atmIsEnemy(p.hit_to)) { d*=0.2f; }
                atmGetFluidModule()->addFluid(PSET_SPHERE_BULLET, p.computeTransformMatrix());
                atmCall(p.hit_to, damage, d);
                atmCall(p.hit_to, pulse, atmArgs(p.pos, p.vel*10000.0f));
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
        if(atmGetConfig()->lighting_level<atmE_Lighting_High) {
            flash  = glow * 0.7f;
        }

        PSetInstance inst;
        inst.diffuse = diffuse;
        inst.glow = glow;
        inst.flash = flash;
        inst.elapsed = 0.0f;
        inst.appear_radius = 10000.0f;
        each(m_bullets, [&](BulletData &p){
            inst.transform = inst.rotate = p.computeTransformMatrix();
            atmGetFluidPass()->addParticles(PSET_SPHERE_BULLET, inst);
        });
        if(atmGetConfig()->lighting_level>=atmE_Lighting_High) {
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

    template<class F>
    void eachBullets(const F &f) { each(m_bullets, f); }
};
atmExportClass(BulletManager)



class NeedleManager
{
private:
public:
};



istSerializeBlockImpl(BulletModule,
    istSerializeBase(super)
    istSerialize(m_lasers)
    istSerialize(m_bullets)
    istSerialize(m_managers)
)
atmExportClass(BulletModule)

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

void BulletModule::handleStateQuery( EntitiesQueryContext &ctx )
{
    m_bullets->eachBullets([&](const BulletData &b){
        ctx.bullets.push_back(vec2(b.pos));
    });
    m_lasers->eachLasers([&](const Laser *l){
        ctx.lasers.push_back(vec3(vec2(l->getPosition()), 1.0f));
        ctx.lasers.push_back(vec3(vec2(l->getPosition()+l->getDirection()*2.0f), 0.0f));
    });
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
    return m_lasers->getLaser(v);
}


} // namespace atm
