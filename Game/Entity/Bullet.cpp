#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"

namespace atm {

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
        atmPlaySE(SE_CHANNEL2, SE_EXPLOSION2, getPosition(), true);
        atmDeleteEntity(getHandle());
    }
};
atmImplementEntity(Bullet_Simple);
atmExportClass(atm::Bullet_Simple);

} // namespace atm
