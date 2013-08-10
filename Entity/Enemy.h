#ifndef atm_Game_Entity_Enemy_h
#define atm_Game_Entity_Enemy_h

#include "EntityCommon.h"
#include "EntityTemplate.h"
#include "Routine.h"

namespace atm {


class Attr_Life
{
private:
    vec4    m_flash_color;
    float32 m_life;
    float32 m_delta_damage;

    istSerializeBlock(
        istSerialize(m_flash_color)
        istSerialize(m_life)
        istSerialize(m_delta_damage)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(getLife)
            atmECall(setLife)
            atmECall(damage)
            atmECall(destroy)
            atmECall(instruct)
        )
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        wdmAddNode(path+"/m_health", &m_life);
        wdmAddNode(path+"/damage()", &Attr_Life::damage, this);
        wdmAddNode(path+"/destroy()", &Attr_Life::destroy, this);
    }
    )

public:
    Attr_Life() : m_life(1.0f), m_delta_damage(0.0f)
    {
    }

    float32 getLife() const         { return m_life; }
    void    setLife(float32 v)      { m_life=v; }

    virtual void damage(float32 d)
    {
        if(m_life > 0.0f) {
            m_life -= d;
            m_delta_damage += d;
            if(m_life <= 0.0f) {
                destroy();
            }
        }
    }

    virtual void destroy() {}
    virtual void instruct(const vec3 &pos, EntityHandle e) {}
};


class IRoutine;

template<class Attributes>
class Breakable : public EntityTemplate<Attributes>
{
typedef EntityTemplate<Attributes> super;
private:
    vec4        m_damage_color;
    IRoutine    *m_routine;
    float32     m_life;
    float32     m_delta_damage;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_damage_color)
        istSerialize(m_routine)
        istSerialize(m_life)
        istSerialize(m_delta_damage)
    )

public:
    atmECallBlock(
        atmECallDelegate(m_routine)
        atmMethodBlock(
            atmECall(isDead)
            atmECall(getLife)
            atmECall(setLife)
            atmECall(setRoutine)
            atmECall(damage)
        )
        atmECallSuper(super)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        super::addDebugNodes(path);
        wdmAddNode(path+"/m_health", &m_life);
        wdmAddNode(path+"/damage()", &Breakable::damage, this);
        wdmAddNode(path+"/destroy()", &Breakable::destroy, this);
    }
    )

    atmJsonizeBlock(
        atmJsonizeSuper(super)
        atmJsonizeMember(m_life, getLife, setLife)
    )

public:
    Breakable()
    : m_routine(nullptr), m_life(1.0f), m_delta_damage(0.0f)
    {}

    ~Breakable()
    {
        setRoutine(RCID_Null);
    }

    bool        isDead() const          { return m_life<=0.0f; }
    float32     getLife() const         { return m_life; }
    IRoutine*   getRoutine()            { return m_routine; }
    const vec4& getDamageColor() const  { return m_damage_color; }
    void        damage(float32 d)       { m_delta_damage += d; }

    void setLife(float32 v)       { m_life=v; }
    void setRoutine(RoutineClassID rcid)
    {
        if(m_routine) { m_routine->finalize(); }
        istSafeDelete(m_routine);
        m_routine = CreateRoutine(rcid);
        if(m_routine) { m_routine->setEntity(this); }
    }

    void finalize() override
    {
        setRoutine(RCID_Null);
    }

    void update(float32 dt) override
    {
        super::update(dt);
        updateRoutine(dt);
        updateDamageFlash();
        updateLife();
    }

    virtual void updateRoutine(float32 dt)
    {
        if(m_routine) { m_routine->update(dt); }
    }

    virtual void updateLife()
    {
        if(!isDead()) {
            m_life -= m_delta_damage;
            m_delta_damage = 0.0f;
            if(m_life <= 0.0f) {
                destroy();
            }
        }
    }

    virtual void updateDamageFlash()
    {
        m_damage_color = vec4();
        if(fmod(super::getPastTime(), 4.0f) < 2.0f) {
            const float32 threthold1 = 0.05f;
            const float32 threthold2 = 1.0f;
            const float32 threthold3 = 10.0f;
            if(m_delta_damage < threthold1) {
            }
            else if(m_delta_damage < threthold2) {
                float32 d = m_delta_damage - threthold1;
                float32 r = threthold2 - threthold1;
                m_damage_color = vec4(d/r, d/r, 0.0f, 0.0f);
            }
            else if(m_delta_damage) {
                float32 d = m_delta_damage - threthold2;
                float32 r = threthold3 - threthold2;
                m_damage_color = vec4(1.0f, stl::max<float32>(1.0f-d/r, 0.0f), 0.0f, 0.0f);
            }
            m_damage_color *= 0.25f;
        }
    }


    void asyncupdate(float32 dt) override
    {
        asyncupdateRoutine(dt);
    }

    virtual void asyncupdateRoutine(float32 dt)
    {
        if(m_routine) { m_routine->asyncupdate(dt); }
    }

    virtual void destroy()
    {
        //atmDeleteEntity(getHandle());
    }

    virtual void eventFluid(const FluidMessage *m)
    {
        addBloodstain(getInvTransformMatrix(), (vec4&)m->position);
        m_delta_damage += glm::length((const vec3&)m->velocity)*0.01f;
    }


    void draw() override
    {
        PSetInstance inst;
        inst.diffuse = getDiffuseColor();
        inst.glow = getGlowColor();
        inst.flash = getDamageColor();
        inst.elapsed = (float32)getPastTime();
        inst.appear_radius = inst.elapsed * 0.004f;
        inst.transform = getTransformMatrix();
        atmGetFluidPass()->addParticles(getModel(), inst);
        atmGetBloodStainPass()->addBloodstainParticles(getTransformMatrix(), getBloodStainParticles(), getNumBloodstainParticles());
    }
};

} // namespace atm
#endif // atm_Game_Entity_Enemy_h
