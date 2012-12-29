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


class Player
    : public Breakable
    , public TAttr_TransformMatrix< TAttr_RotateSpeed<Attr_DoubleAxisRotation> >
{
typedef Player this_t;
typedef Breakable super;
typedef TAttr_TransformMatrix< TAttr_RotateSpeed<Attr_DoubleAxisRotation> > transform;
private:
    static const PSET_RID pset_id = PSET_SPHERE_SMALL;

    vec4 m_vel;
    int32 m_cooldown;
    Attr_Collision m_collision;
    Attr_Collision m_barrier;

    vec4 m_lightpos[1];
    vec4 m_lightvel[1];

public:
    istIntrospectionBlock(
        istName(Player)
        istSuper(super)
        istSuper(transform)
        istMember(m_vel)
        istMember(m_cooldown)
        istMember(m_collision)
        istMember(m_barrier)
        istMember(m_lightpos)
        istMember(m_lightvel)
    )
    atomicECallBlock(
        atomicECallSuper(super)
        atomicECallSuper(transform)
    )
    atomicEQueryBlock(
        atomicEQuerySuper(super)
        atomicEQuerySuper(transform)
    )

public:
    Player() : m_cooldown(0)
    {
    }

    virtual void initialize()
    {
        super::initialize();
        m_collision.initializeCollision(getHandle());
        m_collision.setCollisionShape(CS_SPHERE);
        m_barrier.initializeCollision(0);
        m_barrier.setCollisionShape(CS_SPHERE);
        m_barrier.setCollisionFlags(CF_SPH_SENDER);

        setHealth(500.0f);
        setAxis1(GenRandomUnitVector3());
        setAxis2(GenRandomUnitVector3());
        setRotateSpeed1(1.4f);
        setRotateSpeed2(1.4f);

        for(uint32 i=0; i<_countof(m_lightpos); ++i) {
            m_lightpos[i] = GenRandomVector3() * 1.0f;
            m_lightpos[i].z = std::abs(m_lightpos[i].z);
        }
    }

    void move()
    {
        m_cooldown = stl::max<int32>(0, m_cooldown-1);

        vec4 move = vec4(atomicGetIngameInputs()->getMove()*0.01f, 0.0f, 0.0f);
        if(m_cooldown==0 && atomicGetIngameInputs()->isButtonTriggered(0)) {
            m_vel += move * 2.0f;
            m_cooldown = 10;
        }

        vec4 pos = getPosition();
        pos += move;
        pos += m_vel;
        pos.z = 0.0f;
        setPosition(pos);

        m_vel *= 0.96f;
    }

    virtual void update(float32 dt)
    {
        super::update(dt);
        {
            const vec4 &pos = getPosition();
            psym::PointForce force;
            force.x = pos.x;
            force.y = pos.y;
            force.z = pos.z;
            force.strength = 2.0f;
            atomicGetSPHManager()->addForce(force);
        }
        if(atomicGetSPHManager()->getNumParticles()<10000) {
            psym::Particle particles[16];
            for(size_t i=0; i<_countof(particles); ++i) {
                vec4 rd = glm::normalize(vec4(atomicGenRandFloat()-0.5f, atomicGenRandFloat()-0.5f, 0.0f, 0.0f));
                istAlign(16) vec4 pos = getPosition() + (rd * (atomicGenRandFloat()*0.2f+0.4f));
                psym::simdvec4 poss = (psym::simdvec4&)pos;
                particles[i].position = poss;
                particles[i].velocity = _mm_set1_ps(0.0f);
            }
            atomicGetSPHManager()->addFluid(&particles[0], _countof(particles));
        }
    }

    void asyncupdate(float32 dt)
    {
        super::asyncupdate(dt);

        move();
        updateLights();

        transform::updateRotate(dt);
        transform::updateTransformMatrix();
        m_collision.updateCollisionByParticleSet(pset_id, getTransform(), 0.5f);
        m_barrier.updateCollisionByParticleSet(pset_id, getTransform(), 3.0f);
    }

    void updateLights()
    {
        vec4 diff[4] = {
            vec4( 0.0f, 0.0f, 0.0f, 0.0f),
            vec4(-0.4f, 0.4f, 0.0f, 0.0f),
            vec4(-0.4f,-0.4f, 0.0f, 0.0f),
            vec4( 0.4f,-0.4f, 0.0f, 0.0f),
        };
        for(uint32 i=0; i<_countof(m_lightpos); ++i) {
            vec4 &pos = m_lightpos[i];
            vec4 &vel = m_lightvel[i];
            vel *= 0.985f;
            vel += glm::normalize(getPosition()+diff[i]-pos) * 0.005f;
            pos += vel;
            pos.z = 0.5f;
        }
    }

    virtual void draw()
    {
        {
            PointLight l;
            l.setPosition(getPosition()+vec4(0.0f, 0.0f, 0.3f, 0.0f));
            l.setColor(vec4(0.1f, 0.2f, 1.0f, 1.0f));
            l.setRadius(1.0f);
            atomicGetLights()->addLight(l);
        }
        for(uint32 i=0; i<_countof(m_lightpos); ++i) {
            vec4 &pos = m_lightpos[i];
            PointLight l;
            l.setPosition(pos);
            l.setColor(vec4(0.3f, 0.6f, 0.6f, 1.0f) + vec4(sinf(pos.x), sinf(pos.y), cosf(pos.x+pos.y), 0.0f)*0.1f);
            l.setRadius(1.2f);
            atomicGetLights()->addLight(l);
        }
        {
            PSetInstance inst;
            inst.diffuse = vec4(0.6f, 0.6f, 0.6f, 50.0f);
            inst.glow = vec4(0.2f, 0.0f, 1.0f, 0.0f);
            inst.flash = vec4();
            inst.elapsed = (float32)getPastFrame();
            atomicGetSPHRenderer()->addPSetInstance(pset_id, getTransform(), inst);
        }
        //{
        //    IndivisualParticle particles;
        //    particles.position = getPosition()+vec4(0.3f, 0.3f, 0.05f, 0.0f);
        //    particles.color = vec4(0.6f, 0.6f, 0.6f, 50.0f);
        //    particles.glow = vec4(0.15f, 0.15f, 0.3f, 1.0f);
        //    particles.scale = 3.0f;
        //    atomicGetParticleRenderer()->addParticle(&particles, 1);
        //}
    }

    virtual void destroy()
    {
        atomicGetSPHManager()->addFluid(pset_id, getTransform());
        atomicPlaySE(SE_CHANNEL5, SE_EXPLOSION5, getPosition(), true);
        super::destroy();
    }

    virtual void eventCollide(const CollideMessage *m)
    {
        if( m->cfrom==m_barrier.getCollisionHandle() ||
            m->cto==m_barrier.getCollisionHandle()) { return; }

        vec4 v = m->direction * m->direction.w * 0.2f;
        m_vel += v;
        m_vel.z = 0.0f;
        m_vel.w = 0.0f;

        damage(m->direction.w * 100.0f);
    }
};

atomicImplementEntity(Player, ECID_Player);

} // namespace atomic
