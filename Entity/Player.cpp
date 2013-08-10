#include "stdafx.h"
#include "types.h"
#include "Util.h"
#include "Sound/AtomicSound.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/FluidModule.h"
#include "Game/CollisionModule.h"
#include "Game/Message.h"
#include "Enemy.h"

namespace atm {

class IWeaponry
{
private:
    EntityHandle m_owner;

    istSerializeBlock(
        istSerialize(m_owner)
    )

public:
    IWeaponry() : m_owner(0) {}
    virtual ~IWeaponry() {}
    virtual void update(float32 dt){}
    virtual void asyncupdate(float32 dt) {}
    virtual void draw() {}

    void         setOwner(const IEntity *e) { m_owner = e ? e->getHandle() : 0; }
    IEntity*     getOwner() const { return atmGetEntity(m_owner); }
    EntityHandle getOwnerHandle() const { return m_owner; }
};


class Booster : public IWeaponry
{
typedef IWeaponry super;
private:
    EntityHandle m_barrier;
    float32 m_cooldown;
    bool m_fixed;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_barrier)
        istSerialize(m_cooldown)
        istSerialize(m_fixed)
    )

public:
    Booster() : m_barrier(0), m_cooldown(0), m_fixed(false)
    {
    }

    void update(float32 dt) override
    {
        vec3 center_force;
        IEntity *barrier = atmGetEntity(m_barrier);
        IEntity *owner = getOwner();
        if(atmGetIngameInputs().isButtonTriggered(1)) {
            m_fixed = !m_fixed;
        }
        if(!barrier) {
            IEntity *e = atmCreateEntityT(Barrier);
            m_barrier = e->getHandle();
            atmCall(e, setOwner, getOwnerHandle());
            atmQuery(owner, getPosition, center_force);
        }
        else {
            if(barrier && owner && !m_fixed) {
                vec3 barrier_pos = atmGetProperty(vec3, owner, getPositionAbs);
                atmCall(barrier, setPosition, barrier_pos);
            }
            atmQuery(barrier, getPosition, center_force);
        }

        {
            psym::PointForce force;
            force.x = center_force.x;
            force.y = center_force.y;
            force.z = center_force.z;
            force.strength = 6.0f;
            atmGetFluidModule()->addForce(force);
        }
    }

    void asyncupdate(float32 dt) override
    {
        m_cooldown = stl::max<float32>(0.0f, m_cooldown-dt);

        IEntity *owner = getOwner();
        if(owner) {
            vec3 move = vec3(atmGetIngameInputs().getMove()*0.01f, 0.0f);
            vec3 pos, vel;
            atmQuery(owner, getPosition, pos);
            atmQuery(owner, getVelocity, vel);
            pos += (move + vel) * dt;
            vel *= 0.96f;
            pos.z = 0.0f;
            if(m_cooldown==0.0f && atmGetIngameInputs().isButtonTriggered(0)) {
                vel += move * 2.0f;
                m_cooldown = 10;
            }
            atmCall(owner, setPosition, pos);
            atmCall(owner, setVelocity, vel);
        }
    }

    void draw() override
    {
    }
};
atmExportClass(Booster);


class Blinker : public IWeaponry
{
typedef IWeaponry super;
private:
    static const int32 max_pos_stack = 3;
    EntityHandle m_barrier;
    std::deque<vec3> m_blink_pos;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_barrier)
        istSerialize(m_blink_pos)
    )

public:
    Blinker() : m_barrier(0), m_blink_pos()
    {
    }

    void update(float32 dt) override
    {
        vec3 center_force;
        IEntity *barrier = atmGetEntity(m_barrier);
        IEntity *owner = getOwner();
        if(!barrier) {
            IEntity *e = atmCreateEntityT(Barrier);
            m_barrier = e->getHandle();
            atmCall(e, setOwner, getOwnerHandle());
            atmQuery(owner, getPosition, center_force);
        }
        else {
            if(barrier && owner) {
                vec3 barrier_pos;
                atmQuery(owner, getPosition, barrier_pos);
                atmCall(barrier, setPosition, barrier_pos);
            }
            atmQuery(barrier, getPosition, center_force);
        }

        {
            psym::PointForce force;
            force.x = center_force.x;
            force.y = center_force.y;
            force.z = center_force.z;
            force.strength = 6.0f;
            atmGetFluidModule()->addForce(force);
        }
    }

    void asyncupdate(float32 dt) override
    {
        IEntity *owner = getOwner();
        if(owner) {
            vec3 move = vec3(atmGetIngameInputs().getMove()*0.01f, 0.0f);
            vec3 pos, vel;
            atmQuery(owner, getPosition, pos);
            atmQuery(owner, getVelocity, vel);

            pos += move*dt;
            if(atmGetIngameInputs().isButtonTriggered(0)) {
                m_blink_pos.push_back(pos);
                if(m_blink_pos.size()>max_pos_stack) {
                    m_blink_pos.pop_front();
                }
            }
            if(atmGetIngameInputs().isButtonTriggered(1)) {
                if(!m_blink_pos.empty()) {
                    pos = m_blink_pos.back();
                    m_blink_pos.pop_back();
                    // todo: エフェクト出すとか
                }
            }

            pos += vel*dt;
            vel *= 0.96f;
            pos.z = 0.0f;
            atmCall(owner, setPosition, pos);
            atmCall(owner, setVelocity, vel);
        }
    }

    void draw() override
    {
        each(m_blink_pos, [](const vec3 &){
            // todo: 位置表示
        });
    }
};
atmExportClass(Blinker);


class Catapult : public IWeaponry
{
typedef IWeaponry super;
private:
    float32 m_energy;
    float32 m_cooldown;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_energy)
        istSerialize(m_cooldown)
    )

public:
    Catapult() : m_energy(), m_cooldown()
    {
    }

    void catapult()
    {
        // todo:
    }

    void update(float32 dt) override
    {
        vec3 center_force;
        IEntity *owner = getOwner();
        atmQuery(owner, getPosition, center_force);
        {
            psym::PointForce force;
            force.x = center_force.x;
            force.y = center_force.y;
            force.z = center_force.z;
            force.strength = 6.0f;
            atmGetFluidModule()->addForce(force);
        }
        if(atmGetIngameInputs().isButtonTriggered(1)) {
            catapult();
        }
    }

    void asyncupdate(float32 dt) override
    {
        m_cooldown = stl::max<float32>(0.0f, m_cooldown-dt);

        IEntity *owner = getOwner();
        if(owner) {
            vec3 move = vec3(atmGetIngameInputs().getMove()*0.01f, 0.0f);
            vec3 pos, vel;
            atmQuery(owner, getPosition, pos);
            atmQuery(owner, getVelocity, vel);
            pos += (move + vel) * dt;
            vel *= 0.96f;
            pos.z = 0.0f;
            if(m_cooldown==0.0f && atmGetIngameInputs().isButtonTriggered(0)) {
                vel += move * 2.0f;
                m_cooldown = 10;
            }
            atmCall(owner, setPosition, pos);
            atmCall(owner, setVelocity, vel);
        }
    }

    void draw() override
    {
    }
};
atmExportClass(Catapult);


static const float32 g_player_life_max = 150.0f;
static const float32 g_player_life_regen    = 150.0f/1200.0f;

class Player : public Breakable<Entity_AxisRotationI>
{
typedef Breakable<Entity_AxisRotationI> super;
private:
    static const PSET_RID pset_id = PSET_SPHERE_SMALL;
    enum State {
        State_Normal,
        State_Dead,
    };

    IWeaponry   *m_weapon;
    vec3        m_vel;
    vec3        m_lightpos[1];
    vec3        m_lightvel[1];
    State       m_state;
    float32     m_life_max;
    float32     m_life_regen;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_weapon)
        istSerialize(m_vel)
        istSerialize(m_lightpos)
        istSerialize(m_lightvel)
        istSerialize(m_state)
        istSerialize(m_life_max)
        istSerialize(m_life_regen)
    )

public:
    enum {
        Weponry_Booster,
        Weponry_Blinker,
        Weponry_Catapult,
    };

    atmECallBlock(
        atmECallSuper(super)
        atmMethodBlock(
            atmECall(getVelocity)
            atmECall(setVelocity)
            atmECall(setWeapon)
        )
    )

public:
    Player()
        : m_vel()
        , m_weapon(nullptr)
        , m_state(State_Normal)
        , m_life_max(g_player_life_max)
        , m_life_regen(g_player_life_regen)
    {
        wdmScope(
            wdmString path = wdmFormat("Player/0x%p", this);
            super::addDebugNodes(path);
            transform::addDebugNodes(path);
            wdmAddNode(path+"/setWeapon()", &Player::setWeapon, this);
        )
    }

    ~Player()
    {
        istSafeDelete(m_weapon);
        wdmEraseNode(wdmFormat("Player/0x%p", this));
    }

    const vec3& getVelocity() const { return m_vel; }
    void setVelocity(const vec3 &v) { m_vel=v; }

    void setWeapon(int32 id) // id: Weponry_Barrier, etc
    {
        IWeaponry *old_weapon = m_weapon;
        switch(id) {
        case Weponry_Booster:
            m_weapon = istNew(Booster)();
            break;
        case Weponry_Blinker:
            m_weapon = istNew(Blinker)();
            break;
        case Weponry_Catapult:
            m_weapon = istNew(Catapult)();
            break;
        default:
            istAssert(false && "unknown weapon type");
            return;
        }
        m_weapon->setOwner(this);
        istSafeDelete(old_weapon);
    }

    void initialize() override
    {
        super::initialize();

        setWeapon(Weponry_Booster);

        initializeCollision(getHandle());
        setCollisionShape(CS_Sphere);
        getCollisionSphere().pos_r.w = 0.125f*0.5f;

        setLife(m_life_max);
        setAxis1(GenRandomUnitVector3());
        setAxis2(GenRandomUnitVector3());
        setRotateSpeed1(1.4f);
        setRotateSpeed2(1.4f);

        for(uint32 i=0; i<_countof(m_lightpos); ++i) {
            m_lightpos[i] = GenRandomVector3() * 1.0f;
            m_lightpos[i].z = std::abs(m_lightpos[i].z);
        }
    }

    void update(float32 dt) override
    {
        super::update(dt);
        if(m_state==State_Dead) {
            atmDeleteEntity(getHandle());
            return;
        }

        setLife(std::min<float32>(getLife()+m_life_regen, m_life_max));
        if(m_weapon) {m_weapon->update(dt); }
        {
            vec3 pos = getPosition();
            float32 l = PSYM_GRID_SIZE*0.5f*0.95f;
            pos.x = clamp(pos.x, -l, l);
            pos.y = clamp(pos.y, -l, l);
            pos.z = 0.0f;
            setPosition(pos);
        }

        // 流体パーティクルが 10000 以下なら追加
        if(atmGetFluidModule()->getNumParticles()<10000) {
            psym::Particle particles[16];
            for(size_t i=0; i<_countof(particles); ++i) {
                vec4 rd = glm::normalize(vec4(atmGenRandFloat()-0.5f, atmGenRandFloat()-0.5f, 0.0f, 0.0f));
                istAlign(16) vec4 pos = vec4(getPosition(), 1.0f) + (rd * (atmGenRandFloat()*0.2f+0.4f));
                psym::simdvec4 poss = (psym::simdvec4&)pos;
                particles[i].position = poss;
                particles[i].velocity = _mm_set1_ps(0.0f);
            }
            atmGetFluidModule()->addFluid(&particles[0], _countof(particles));
        }
    }

    void asyncupdate(float32 dt) override
    {
        super::asyncupdate(dt);
        if(m_weapon) {m_weapon->asyncupdate(dt); }

        updateLights();

        transform::updateRotate(dt);
        transform::updateTransformMatrix();
        collision::updateCollisionByParticleSet(pset_id, getTransformMatrix(), vec3(0.5f));
    }

    void updateLights()
    {
        vec3 diff[] = {
            vec3( 0.0f, 0.0f, 0.0f),
            vec3(-0.4f, 0.4f, 0.0f),
            vec3(-0.4f,-0.4f, 0.0f),
            vec3( 0.4f,-0.4f, 0.0f),
        };
        for(uint32 i=0; i<_countof(m_lightpos); ++i) {
            vec3 &pos = m_lightpos[i];
            vec3 &vel = m_lightvel[i];
            vel *= 0.985f;
            vel += glm::normalize(getPosition()+diff[i]-pos) * 0.005f;
            pos += vel;
            pos.z = 0.5f;
        }
    }

    void draw() override
    {
        {
            PointLight l;
            l.setPosition(getPosition()+vec3(0.0f, 0.0f, 0.3f));
            l.setColor(vec4(0.3f, 0.2f, 1.0f, 1.0f));
            l.setRadius(1.0f);
            atmGetLightPass()->addLight(l);
        }
        for(uint32 i=0; i<_countof(m_lightpos); ++i) {
            vec3 &pos = m_lightpos[i];
            PointLight l;
            l.setPosition(pos);
            l.setColor(vec4(0.45f, 0.45f, 0.6f, 1.0f) + vec4(sinf(pos.x), sinf(pos.y), cosf(pos.x+pos.y), 0.0f)*0.1f);
            l.setRadius(1.2f);
            atmGetLightPass()->addLight(l);
        }
        {
            PSetInstance inst;
            inst.diffuse = vec4(0.6f, 0.6f, 0.6f, 50.0f);
            inst.glow = vec4(0.2f, 0.0f, 1.0f, 0.0f);
            inst.flash = vec4();
            inst.elapsed = (float32)getPastTime();
            inst.appear_radius = 1000.0f;
            inst.transform = inst.rotate = getTransformMatrix();
            atmGetFluidPass()->addParticles(pset_id, inst);
        }
        //{
        //    IndivisualParticle particles;
        //    particles.position = getPosition()+vec4(0.3f, 0.3f, 0.05f, 0.0f);
        //    particles.color = vec4(0.6f, 0.6f, 0.6f, 50.0f);
        //    particles.glow = vec4(0.15f, 0.15f, 0.3f, 1.0f);
        //    particles.scale = 3.0f;
        //    atmGetParticlePass()->addParticle(&particles, 1);
        //}
    }

    void destroy() override
    {
        atmGetFluidModule()->addFluid(pset_id, getTransformMatrix());
        atmPlaySE(SE_CHANNEL5, SE_EXPLOSION5, getPosition(), true);
        m_state = State_Dead;
    }

    //void eventFluid(const FluidMessage *m) override
    //{
    //    addBloodstain(getInvTransformMatrix(), (vec4&)m->position);
    //    damage(glm::length((const vec3&)m->velocity)*0.015f);
    //}

    void eventCollide(const CollideMessage *m) override
    {
        // 押し返し
        vec3 v = vec3(m->direction * (m->direction.w * 0.2f));
        m_vel += v;
        m_vel.z = 0.0f;

        damage(m->direction.w * 100.0f);
    }
};
atmImplementEntity(Player);
atmExportClass(Player);




class Barrier : public EntityTemplate<Entity_Translate>
{
typedef EntityTemplate<Entity_Translate>    super;
private:
    EntityHandle    m_owner;
    float32         m_life;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_owner)
        istSerialize(m_life)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
        atmMethodBlock(
        atmECall(setOwner)
        atmECall(getOwner)
        )
    )

public:
    Barrier() : m_owner(0), m_life(100.0f)
    {
    }

    void            setOwner(EntityHandle v)    { m_owner=v; }
    EntityHandle    getOwner() const            { return m_owner; }

    virtual void initialize()
    {
        initializeCollision(getHandle());
        setCollisionShape(CS_Sphere);
        setCollisionFlags(CF_SPH_Sender);
        getCollisionSphere().pos_r.w = 0.125f*3.0f;
    }

    virtual void finalize()
    {
    }

    virtual void update(float32 dt)
    {
    }

    virtual void asyncupdate(float32 dt)
    {
        super::update(dt);

        transform::updateTransformMatrix();
        updateCollision(transform::getTransformMatrix());
    }

    virtual void draw()
    {
        const vec3 &pos = getPosition();
        {
            PointLight l;
            l.setPosition(pos+vec3(0.0f, 0.0f, 0.3f));
            l.setColor(vec4(0.3f, 0.2f, 1.0f, 1.0f));
            l.setRadius(1.0f);
            atmGetLightPass()->addLight(l);
        }
        {
            float32 radius = 0.125f*3.0f;
            PassForward_Generic::InstanceParams params;
            params.transform *= glm::translate(mat4(), pos);
            params.transform *= glm::scale(mat4(), vec3(radius));
            atmGetForwardPass()->drawModel(SH_BARRIER, MODEL_UNITSPHERE, params);
            //params.params[0] = vec4(radius, 0.1f, 1.0f, 1.0f);
            //atmGetForwardPass()->drawModel(SH_FEEDBACK_BLUR, MODEL_UNITSPHERE, params);
        }
    }
};
atmImplementEntity(Barrier);
atmExportClass(Barrier);

} // namespace atm
