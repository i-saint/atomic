#include "stdafx.h"
#include "types.h"
#include "Util.h"
#include "Sound/AtomicSound.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "Game/Collision.h"
#include "Game/Message.h"
#include "Enemy.h"

namespace atomic {

class Bullet_Laser
    : public IEntity
    , public TAttr_TransformMatrix< Attr_Translate >
{
typedef Bullet_Laser this_t;
typedef IEntity super;
typedef TAttr_TransformMatrix< Attr_Translate > transform;
public:
    struct LaserParticle
    {
        vec3 initial_pos;
        float32 elapsed;
    };

private:
    stl::vector<LaserParticle> m_particles;
    vec4 m_dir;
    float32 m_speed;
    EntityHandle m_owner;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerialize(m_particles)
        istSerialize(m_dir)
        istSerialize(m_speed)
        istSerialize(m_owner)
        )

public:
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setOwner)
        atomicECall(setDirection)
        atomicECall(setSpeed)
        )
        atomicECallSuper(super)
        atomicECallSuper(transform)
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getOwner)
        atomicEQuery(getDirection)
        atomicEQuery(getSpeed)
        )
        atomicEQuerySuper(super)
        atomicEQuerySuper(transform)
    )

public:
    Bullet_Laser() : m_speed(0.1f)
    {
    }

    EntityHandle getOwner() const   { return m_owner; }
    const vec4& getDirection() const{ return m_dir; }
    float32 getSpeed() const        { return m_speed; }

    void setOwner(EntityHandle v)   { m_owner=v; }
    void setDirection(const vec4 &v){ m_dir=v; }
    void setSpeed(float32 v)        { m_speed=v; }


    void initialize()
    {
    }

    virtual void update(float32 dt)
    {
        super::update(dt);
    }

    virtual void asyncupdate(float32 dt)
    {
        super::asyncupdate(dt);
        transform::updateTransformMatrix();
    }

    virtual void draw()
    {
    }
};
atomicImplementEntity(Bullet_Laser);
atomicExportClass(atomic::Bullet_Laser);
atomicSerializeRaw(atomic::Bullet_Laser::LaserParticle);


class Bullet_Particle
    : public IEntity
    , public TAttr_TransformMatrix< Attr_Translate >
    , public Attr_Collision
    , public Attr_MessageHandler
{
typedef Bullet_Particle this_t;
typedef IEntity super;
typedef TAttr_TransformMatrix< Attr_Translate > transform;
typedef Attr_Collision collision;
typedef Attr_MessageHandler mhandler;
private:
    vec4            m_vel;
    EntityHandle    m_owner;
    float32         m_power;
    int32           m_past_frame;
    int32           m_lifetime;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerializeBase(collision)
        istSerializeBase(mhandler)
        istSerialize(m_vel)
        istSerialize(m_owner)
        istSerialize(m_power)
        istSerialize(m_past_frame)
        istSerialize(m_lifetime)
        )

public:
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setOwner)
        atomicECall(setVelocity)
        atomicECall(setPower)
        )
        atomicECallSuper(super)
        atomicECallSuper(transform)
        atomicECallSuper(mhandler)
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getOwner)
        atomicEQuery(getVelocity)
        atomicEQuery(getPower)
        )
        atomicEQuerySuper(super)
        atomicEQuerySuper(transform)
        atomicEQuerySuper(collision)
    )

public:
    Bullet_Particle() : m_owner(0), m_power(50.0f), m_past_frame(0), m_lifetime(600) {}

    EntityHandle getOwner() const   { return m_owner; }
    const vec4& getVelocity() const { return m_vel; }
    int32 getPastFrame() const      { return m_past_frame; }
    int32 getLifeTime() const       { return m_lifetime; }
    float32 getPower() const        { return m_power; }

    void setOwner(EntityHandle v)   { m_owner=v; }
    void setVelocity(const vec4 &v) { m_vel=v; }
    void setPower(float32 v)        { m_power=v; }


    void initialize()
    {
        collision::initializeCollision(getHandle());
        setCollisionShape(CS_Sphere);
        setCollisionFlags(CF_Receiver | CF_SPH_Sender);
    }

    virtual void update(float32 dt)
    {
        super::update(dt);

        ++m_past_frame;
        if(m_past_frame==m_lifetime) {
            atomicDeleteEntity(getHandle());
            return;
        }
    }

    virtual void asyncupdate(float32 dt)
    {
        super::asyncupdate(dt);
        move();
        transform::updateTransformMatrix();
        collision::updateCollisionAsSphere(getTransform(), 0.015f);
    }

    void move()
    {
        vec4 pos = getPosition();
        pos += getVelocity();
        setPosition(pos);
    }

    virtual void draw()
    {
        {
            IndivisualParticle particles;
            particles.position = getPosition();
            particles.color = vec4(0.6f, 0.3f, 0.3f, 50.0f);
            particles.glow = vec4(0.45f, 0.15f, 0.15f, 1.0f);
            particles.scale = 1.5f;
            atomicGetParticleRenderer()->addParticle(&particles, 1);
        }
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        if(m->from == getOwner()) { return; }

        if(IEntity *e=atomicGetEntity(m->from)) {
            atomicCall(e, damage, m_power);
        }
        //atomicGetSPHManager()->addFluid(getModel(), getTransform());
        //atomicPlaySE(SE_CHANNEL2, SE_EXPLOSION2, getPosition(), true);
        atomicDeleteEntity(getHandle());
    }
};
atomicImplementEntity(Bullet_Particle);
atomicExportClass(atomic::Bullet_Particle);


class Bullet_Simple
    : public IEntity
    , public TAttr_TransformMatrix< TAttr_RotateSpeed<Attr_DoubleAxisRotation> >
    , public Attr_ParticleSet
    , public Attr_Collision
    , public Attr_MessageHandler
{
typedef Bullet_Simple this_t;
typedef IEntity super;
typedef TAttr_TransformMatrix< TAttr_RotateSpeed<Attr_DoubleAxisRotation> > transform;
typedef Attr_ParticleSet model;
typedef Attr_Collision collision;
typedef Attr_MessageHandler mhandler;
private:
    vec4            m_vel;
    EntityHandle    m_owner;
    float32         m_power;
    int32           m_past_frame;
    int32           m_lifetime;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerializeBase(model)
        istSerializeBase(collision)
        istSerializeBase(mhandler)
        istSerialize(m_vel)
        istSerialize(m_owner)
        istSerialize(m_power)
        istSerialize(m_past_frame)
        istSerialize(m_lifetime)
        )

public:
    atomicECallBlock(
        atomicMethodBlock(
        atomicECall(setOwner)
        atomicECall(setVelocity)
        atomicECall(setPower)
        )
        atomicECallSuper(super)
        atomicECallSuper(transform)
        atomicECallSuper(model)
        atomicECallSuper(mhandler)
    )
    atomicEQueryBlock(
        atomicMethodBlock(
        atomicEQuery(getOwner)
        atomicEQuery(getVelocity)
        atomicEQuery(getPower)
        )
        atomicEQuerySuper(super)
        atomicEQuerySuper(transform)
        atomicEQuerySuper(model)
        atomicEQuerySuper(collision)
    )

public:
    Bullet_Simple() : m_owner(0), m_power(50.0f), m_past_frame(0), m_lifetime(600) {}

    EntityHandle getOwner() const   { return m_owner; }
    const vec4& getVelocity() const { return m_vel; }
    int32 getPastFrame() const      { return m_past_frame; }
    int32 getLifeTime() const       { return m_lifetime; }
    float32 getPower() const        { return m_power; }

    void setOwner(EntityHandle v)   { m_owner=v; }
    void setVelocity(const vec4 &v) { m_vel=v; }
    void setPower(float32 v)        { m_power=v; }

    void initialize()
    {
        collision::initializeCollision(getHandle());
        setCollisionShape(CS_Sphere);
        setCollisionFlags(CF_Receiver | CF_SPH_Sender);

        setModel(PSET_SPHERE_SMALL);
        setDiffuseColor(vec4(0.6f, 0.6f, 0.6f, 80.0f));
        setGlowColor(vec4(1.0f, 0.7f, 0.1f, 0.0f));
        //setAxis1(GenRandomUnitVector3());
        //setAxis2(GenRandomUnitVector3());
        setRotateSpeed1(1.5f);
        setRotateSpeed2(1.5f);
    }

    virtual void update(float32 dt)
    {
        super::update(dt);

        ++m_past_frame;
        if(m_past_frame==m_lifetime) {
            atomicDeleteEntity(getHandle());
            return;
        }
    }

    virtual void asyncupdate(float32 dt)
    {
        super::asyncupdate(dt);
        move();
        transform::updateRotate(dt);
        transform::updateTransformMatrix();
        collision::updateCollisionByParticleSet(getModel(), getTransform(), 0.5f);
    }

    void move()
    {
        vec4 pos = getPosition();
        pos += getVelocity();
        setPosition(pos);
    }

    virtual void draw()
    {
        vec4 diffuse= getDiffuseColor();
        vec4 glow   = getGlowColor();
        vec4 light  = glow;

        {
            PointLight l;
            l.setPosition(getPosition() + vec4(0.0f, 0.0f, 0.15f, 1.0f));
            l.setRadius(0.4f);
            l.setColor(light);
            atomicGetLights()->addLight(l);
        }
        PSetInstance inst;
        inst.diffuse = diffuse;
        inst.glow = glow;
        inst.flash = vec4();
        inst.elapsed = (float32)m_past_frame;
        inst.appear_radius = 1000.0f;
        inst.translate = getTransform();
        atomicGetSPHRenderer()->addPSetInstance(getModel(), inst);
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        if(m->from == getOwner()) { return; }

        if(IEntity *e=atomicGetEntity(m->from)) {
            atomicCall(e, damage, m_power);
        }
        atomicGetSPHManager()->addFluid(getModel(), getTransform());
        atomicPlaySE(SE_CHANNEL2, SE_EXPLOSION2, getPosition(), true);
        atomicDeleteEntity(getHandle());
    }
};
atomicImplementEntity(Bullet_Simple);
atomicExportClass(atomic::Bullet_Simple);

} // namespace atomic
