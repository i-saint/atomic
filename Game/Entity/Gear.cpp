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
#include "Routine.h"
#include "Enemy.h"

namespace atm {

class dpPatch GearParts
    : public IEntity
    , public Attr_MessageHandler
    , public Attr_ParticleSet
    , public Attr_Collision
    , public Attr_Bloodstain
    , public TAttr_TransformMatrixI<Attr_Orientation>
{
    typedef IEntity             super;
    typedef Attr_MessageHandler mhandler;
    typedef Attr_ParticleSet    model;
    typedef Attr_Collision      collision;
    typedef Attr_Bloodstain     bloodstain;
    typedef TAttr_TransformMatrixI<Attr_Orientation>    transform;

    EntityHandle m_parent;
    vec3 m_dir;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(model)
        istSerializeBase(collision)
        istSerializeBase(bloodstain)
        istSerializeBase(transform)
        istSerialize(m_parent)
    )

    atmECallBlock(
        atmECallSuper(super)
        atmECallSuper(transform)
        atmECallSuper(collision)
        atmMethodBlock(
            atmECall(getParent)
            atmECall(setParent)
        )
    )

public:
    GearParts() : m_parent(0)
    {
    }

    void accel(const vec3& a)
    {
        IEntity *e = atmGetEntity(getParent());
        if(!e) { return; }

        mat4 pmat; atmQuery(e, getTransform, pmat);
        vec3 dir = vec3( glm::rotateZ(pmat*vec4(getDirection(),0.0f), 90.0f) );
        float32 la = glm::length(a);
        vec3 na = a/la;

        float d = glm::dot(dir, na);
        if(d<-0.5f) {
            atmCall(e, addRotateSpeed, -la);
        }
        else if(d>0.5f) {
            atmCall(e, addRotateSpeed, la);
        }
    }

    EntityHandle    getParent() const           { return m_parent; }
    void            setParent(EntityHandle v)   { m_parent=v; }
    const vec3&     getDirection() const        { return m_dir; }
    void            setDirection(const vec3 &v) { m_dir=v; }
};
atmImplementEntity(GearParts);
atmExportClass(atm::GearParts);


class dpPatch GearBase
    : public IEntity
    , public Attr_MessageHandler
    , public TAttr_TransformMatrix<Attr_Transform>
{
    typedef IEntity             super;
    typedef Attr_MessageHandler mhandler;
    typedef TAttr_TransformMatrix<Attr_Transform>   transform;

    stl::vector<EntityHandle> m_parts;
    float32 m_rot_angle;
    float32 m_rot_speed;
    float32 m_rot_decel;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerializeBase(transform)
        istSerialize(m_parts)
        istSerialize(m_rot_angle)
        istSerialize(m_rot_speed)
        istSerialize(m_rot_decel)
    )

    atmECallBlock(
        atmECallSuper(super)
        atmECallSuper(transform)
    )
public:
    GearBase() : m_rot_angle(0.0f), m_rot_speed(0.0f), m_rot_decel(0.98f)
    {
    }

    ~GearBase()
    {
        eachParts([&](EntityHandle h){ atmDeleteEntity(h); });
    }

    virtual void initialize() override
    {
        transform::setAxis(vec3(0.0f, 0.0f, 1.0f));
    }

    virtual void update(float32 dt) override
    {
        if(SweepDeadEntities(m_parts)==0) {
            atmDeleteEntity(getHandle());
        }
    }

    void updateRotate()
    {
        m_rot_angle += m_rot_speed;
        m_rot_speed *= m_rot_decel;
        transform::setRotate(m_rot_angle);
    }

    void addForce(const vec3& pos, const vec3& force)
    {
        vec3 dir = vec3( glm::rotateZ(glm::normalize(pos-getPosition()), 90.0f) );
        float32 lf = glm::length(force);
        vec3 nf = force/lf;
        float d = glm::dot(dir, nf);
        if(d<-0.5f) {
            addRotateSpeed(-lf);
        }
        else if(d>0.5f) {
            addRotateSpeed(lf);
        }
    }

    void addParts(GearParts *v) { m_parts.push_back(v->getHandle()); }
    float32 getRotateAngle() const  { return m_rot_angle; }
    float32 getRotateSpeed() const  { return m_rot_speed; }
    float32 getRotateDecel() const  { return m_rot_decel; }
    void setRotateAngle(float32 v) { m_rot_angle=v; }
    void setRotateSpeed(float32 v) { m_rot_speed=v; }
    void setRotateDecel(float32 v) { m_rot_decel=v; }
    void addRotateSpeed(float32 v) { m_rot_speed+=v; }
    template<class F> void eachParts(const F &f) { EachEntities(m_parts, f); }
};
atmExportClass(atm::GearBase);


class dpPatch GearSmall : public GearBase
{
    typedef GearBase            super;

    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(GearSmall);
atmExportClass(atm::GearSmall);


class dpPatch GearLarge : public GearBase
{
    typedef GearBase            super;
    typedef Attr_MessageHandler mhandler;

    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(GearLarge);
atmExportClass(atm::GearLarge);

} // namespace atm
