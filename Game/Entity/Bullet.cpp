#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"

namespace atm {

class IBulletManager
{
public:
    virtual ~IBulletManager() {}
    virtual void update(float32 dt) {}
    virtual void asyncupdate(float32 dt) {}
    virtual void draw() {}
};


struct LaserParticle
{
    vec3    pos;
    float32 time;
    EntityHandle hit_to;

    LaserParticle() : time(0.0f), hit_to(0) {}
};
atmSerializeRaw(LaserParticle);

class dpPatch Laser
{
private:
    enum State {
        State_Normal,
        State_Fadeout,
        State_Dead,
    };
    typedef stl::vector<LaserParticle> particles;

    uint32       m_id;
    EntityHandle m_owner;
    State        m_state;
    float32      m_time;
    vec3         m_pos;
    vec3         m_dir;
    particles    m_particles;
    CollisionSet::CollisionContext m_cctx; // serialize 不要

    istSerializeBlock(
        istSerialize(m_id)
        istSerialize(m_owner)
        istSerialize(m_state)
        istSerialize(m_time)
        istSerialize(m_pos)
        istSerialize(m_dir)
        istSerialize(m_particles)
    )

public:
    Laser() : m_id(0), m_owner(0), m_state(State_Normal), m_time(0.0f)
    {
    }

    void update(float32 dt)
    {
        m_time += dt;

        static const float32 speed = 0.1f;
        static const float32 lifetime = 240.0f;
        static const float32 radius = 0.03f;
        std::for_each(m_particles.begin(), m_particles.end(), [&](LaserParticle &p){
            p.time += dt;
        });
        if(m_state==State_Normal) {
            std::for_each(m_particles.begin(), m_particles.end(), [&](LaserParticle &p){
                vec3 pos = p.pos + m_dir*p.time*speed;
                CollisionSphere sphere;
                sphere.pos_r = vec4(pos, radius);
                sphere.bb.bl = vec4(pos-radius, 0.0f);
                sphere.bb.ur = vec4(pos+radius, 0.0f);
                // todo: collision
                // if(collide) {
                //    // add some vfx
                //    // send dame to p.hit_to
                //    p.time=300.0f;
                //}
            });
        }
        m_particles.erase(
            std::remove_if(m_particles.begin(), m_particles.end(), [&](LaserParticle &p){ return p.time>=lifetime; }),
            m_particles.end());
    }

    void fadeout()  { m_state=State_Fadeout; m_time=0.0f; }
    void kill()     { m_state=State_Dead; m_time=0.0f; }
};

class dpPatch LaserManager : public IBulletManager
{
private:
    typedef stl::vector<Laser*> lasers;
    lasers m_lasers;

public:
};


struct SimpleBullet
{
    vec3 pos;
    vec3 vel;
    float32 time;
    EntityHandle owner;
    EntityHandle hit_to;
    uint32 flags;
};
atmSerializeRaw(SimpleBullet);

class dpPatch SimpleBulletManager : public IBulletManager
{
private:
    typedef stl::vector<SimpleBullet> bullets;
    bullets m_bullets;

public:
    SimpleBulletManager()
    {
    }

    void update(float32 dt)
    {
        static const float32 lifetime = 600.0f;
        std::for_each(m_bullets.begin(), m_bullets.end(), [&](SimpleBullet &p){
            p.time += dt;
            p.pos += p.vel;
            // todo: collision
            // if(collide) {
            //    // add some vfx
            //    // send dame to p.hit_to
            //    p.time=300.0f;
            //}
        });
        m_bullets.erase(
            std::remove_if(m_bullets.begin(), m_bullets.end(), [&](SimpleBullet &p){ return p.time>=lifetime; }),
            m_bullets.end());
    }

    void shoot(const vec3 &pos, const vec3 &vel, EntityHandle owner);
};



class dpPatch NeedleBulletManager : public IBulletManager
{
private:
public:
};



class dpPatch Bullet_Needle;

class dpPatch Bullet_Simple : public EntityTemplate<Entity_AxisRotation>
{
typedef EntityTemplate<Entity_AxisRotation> super;
private:
    vec3            m_vel;
    EntityHandle    m_owner;
    float32         m_power;
    int32           m_past_frame;
    int32           m_lifetime;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_vel)
        istSerialize(m_owner)
        istSerialize(m_power)
        istSerialize(m_past_frame)
        istSerialize(m_lifetime)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getOwner)
            atmECall(setOwner)
            atmECall(getVelocity)
            atmECall(setVelocity)
            atmECall(getPower)
            atmECall(setPower)
        )
        atmECallSuper(super)
    )

public:
    Bullet_Simple() : m_owner(0), m_power(2.0f), m_past_frame(0), m_lifetime(600) {}

    EntityHandle getOwner() const   { return m_owner; }
    const vec3& getVelocity() const { return m_vel; }
    int32 getPastFrame() const      { return m_past_frame; }
    int32 getLifeTime() const       { return m_lifetime; }
    float32 getPower() const        { return m_power; }

    void setOwner(EntityHandle v)   { m_owner=v; }
    void setVelocity(const vec3 &v) { m_vel=v; }
    void setPower(float32 v)        { m_power=v; }

    virtual void initialize()
    {
        collision::initializeCollision(getHandle());
        setCollisionShape(CS_Sphere);
        setCollisionFlags(CF_Receiver);

        setModel(PSET_SPHERE_BULLET);
        setDiffuseColor(vec4(0.6f, 0.6f, 0.6f, 80.0f));
        setGlowColor(vec4(2.0f, 1.2f, 0.1f, 0.0f));
        //setAxis1(GenRandomUnitVector3());
        //setAxis2(GenRandomUnitVector3());
        setRotateSpeed1(4.5f);
        setRotateSpeed2(4.5f);
    }

    virtual void update(float32 dt)
    {
        super::update(dt);

        ++m_past_frame;
        if(m_past_frame==m_lifetime) {
            atmDeleteEntity(getHandle());
            return;
        }
    }

    virtual void asyncupdate(float32 dt)
    {
        super::asyncupdate(dt);
        {
            vec3 pos = getPosition();
            pos += getVelocity();
            pos.z = 0.03f;
            setPosition(pos);
        }
        if(m_owner && !atmGetEntity(m_owner)) {
            m_owner = 0;
        }

        transform::updateRotate(dt);
        transform::updateTransformMatrix();
        collision::updateCollisionByParticleSet(getModel(), getTransform(), vec3(0.5f));
    }

    virtual void draw()
    {
        vec4 diffuse= getDiffuseColor();
        vec4 glow   = getGlowColor();
        vec4 light  = glow;
        vec4 flash  = glow * 0.5f;
        if(atmGetConfig()->lighting>=atmE_Lighting_High) {
            PointLight l;
            l.setPosition(getPosition() + vec3(0.0f, 0.0f, 0.10f));
            l.setRadius(0.2f);
            l.setColor(light);
            atmGetLightPass()->addLight(l);
        }
        else {
            flash  = glow * 0.7f;
        }
        PSetInstance inst;
        inst.diffuse = diffuse;
        inst.glow = glow;
        inst.flash = flash;
        inst.elapsed = (float32)m_past_frame;
        inst.appear_radius = 1000.0f;
        inst.translate = getTransform();
        atmGetSPHPass()->addPSetInstance(getModel(), inst);
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        if(m->from==getOwner()) { return; }

        if(IEntity *e=atmGetEntity(m->from)) {
            atmCall(e, damage, m_power);
        }
        atmGetSPHManager()->addFluid(getModel(), getTransform());
        atmDeleteEntity(getHandle());
    }
};
atmImplementEntity(Bullet_Simple);
atmExportClass(Bullet_Simple);

} // namespace atm
