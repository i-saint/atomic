#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Level.h"

namespace atm {

class dpPatch GearParts : public Unbreakable<Entity_Orientation>
{
typedef Unbreakable<Entity_Orientation>  super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
    )

public:
    GearParts()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GearParts/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GearParts()
    {
        wdmEraseNode(wdmFormat("Level/GearParts/0x%p", this));
    }

    virtual void initialize() override
    {
        super::initialize();
        setPivot(vec3(-0.2f, 0.0f, -0.1f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender|CF_SPH_Sender);

        setModel(PSET_CUBE_MEDIUM);
    }


    virtual void eventFluid(const FluidMessage *m) override
    {
        super::eventFluid(m);
        atmCall(getParent(), addForce, atmArgs((const vec3&)m->position, (const vec3&)m->velocity*m->density*0.001f));
    }


    //void accel(const vec3& a)
    //{
    //    IEntity *e = atmGetEntity(getParent());
    //    if(!e) { return; }

    //    mat4 pmat; atmQuery(e, getTransform, pmat);
    //    vec3 dir = vec3( glm::rotateZ(pmat*vec4(getDirection(),0.0f), 90.0f) );
    //    float32 la = glm::length(a);
    //    vec3 na = a/la;

    //    float d = glm::dot(dir, na);
    //    if(d<-0.5f) {
    //        atmCall(e, addRotateSpeed, -la);
    //    }
    //    else if(d>0.5f) {
    //        atmCall(e, addRotateSpeed, la);
    //    }
    //}
};
atmImplementEntity(GearParts);
atmExportClass(GearParts);


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
    float32 m_max_rot_speed;
    float32 m_rot_accel;
    float32 m_rot_decel;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(mhandler)
        istSerializeBase(transform)
        istSerialize(m_parts)
        istSerialize(m_rot_angle)
        istSerialize(m_rot_speed)
        istSerialize(m_max_rot_speed)
        istSerialize(m_rot_accel)
        istSerialize(m_rot_decel)
    )

public:
    atmECallBlock(
        atmMethodBlock(
            atmECall(addForce)
        )
        atmECallSuper(super)
        atmECallSuper(transform)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        transform::addDebugNodes(path);
        wdmAddNode(path+"/m_rot_angle", &m_rot_angle);
        wdmAddNode(path+"/m_rot_speed", &m_rot_speed);
        wdmAddNode(path+"/m_rot_maxspeed", &m_max_rot_speed);
        wdmAddNode(path+"/m_rot_decel", &m_rot_decel);
        wdmAddNode(path+"/m_rot_accel", &m_rot_accel);
    }
    )
public:
    GearBase() : m_rot_angle(0.0f), m_rot_speed(0.0f), m_max_rot_speed(0.5f), m_rot_accel(0.0001f), m_rot_decel(0.99f)
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
    }

    virtual void asyncupdate(float32 dt) override
    {
        updateRotate();
    }

    void updateRotate()
    {
        m_rot_angle += m_rot_speed;
        m_rot_speed *= m_rot_decel;
        transform::setRotate(m_rot_angle);
        transform::updateTransformMatrix();
    }

    void addForce(const vec3 &pos, const vec3 &force)
    {
        float32 dist = glm::length(pos-getPosition());
        vec3 dir = vec3( glm::rotateZ(glm::normalize(pos-getPosition()), 90.0f) );
        float32 lf = glm::length(force);
        vec3 nf = force/lf;
        float d = glm::dot(dir, nf);
        lf *= m_rot_accel * dist;
        if(d<-0.9f) {
            addRotateSpeed(-lf);
        }
        else if(d>0.9f) {
            addRotateSpeed(lf);
        }
    }

    void addParts(GearParts *v)         { m_parts.push_back(v->getHandle()); }
    float32 getRotateAngle() const      { return m_rot_angle; }
    float32 getRotateSpeed() const      { return m_rot_speed; }
    float32 getMaxRotateSpeed() const   { return m_max_rot_speed; }
    float32 getRotateAccel() const      { return m_rot_accel; }
    float32 getRotateDecel() const      { return m_rot_decel; }
    void setRotateAngle(float32 v)      { m_rot_angle=v; }
    void setRotateSpeed(float32 v)      { m_rot_speed=glm::sign(v)*glm::min(glm::abs(v), m_max_rot_speed); }
    void setMaxRotateSpeed(float32 v)   { m_max_rot_speed=v; }
    void setRotateAccel(float32 v)      { m_rot_accel=v; }
    void setRotateDecel(float32 v)      { m_rot_decel=v; }
    void addRotateSpeed(float32 v)      { setRotateSpeed(getRotateSpeed()+v); }
    template<class F> void eachParts(const F &f) { EachEntities(m_parts, f); }
};
atmExportClass(GearBase);


class dpPatch GearSmall : public GearBase
{
typedef GearBase super;

    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    GearSmall()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GearSmall/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GearSmall()
    {
        wdmEraseNode(wdmFormat("Level/GearSmall/0x%p", this));
    }

    virtual void initialize() override
    {
        setMaxRotateSpeed(0.6f);
        setRotateDecel(0.99f);
        setRotateAccel(0.0001f);

        const int32 div = 4;
        const vec4 dir_x(1.0f,0.0f,0.0f,0.0f);
        for(int i=0; i<div; ++i) {
            vec3 dir = vec3(glm::rotateZ(dir_x, 360.0f/div*i));
            GearParts *e = (GearParts*)atmCreateEntity(GearParts);
            e->setScale(vec3(2.5f, 0.75f, 1.0f));
            e->setParent(getHandle());
            e->setOrientation(dir);
            addParts(e);
        }
    }
};
atmImplementEntity(GearSmall);
atmExportClass(GearSmall);


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
atmExportClass(GearLarge);

} // namespace atm
