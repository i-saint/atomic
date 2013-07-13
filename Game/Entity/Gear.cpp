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
        atmMethodBlock(
            atmECall(addForce)
        )
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

    void initialize() override
    {
        super::initialize();
        setPivot(vec3(-0.2f, 0.0f, 0.0f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender|CF_Receiver|CF_SPH_Sender);

        setModel(PSET_CUBE_MEDIUM);
    }


    void eventFluid(const FluidMessage *m) override
    {
        super::eventFluid(m);
        vec3 pos = (const vec3&)m->position;
        vec3 force = (const vec3&)m->velocity;
        addForce(pos, force);
    }

    void eventCollide(const CollideMessage *m) override
    {
        super::eventCollide(m);
        vec3 pos;
        if(atmQuery(m->from, getPosition, pos)) {
            float32 f = 4000.0f;
            if(EntityGetCategory(m->from)==ECA_Obstacle) {
                f = 50000.0f;
            }
            vec3 force = vec3(m->direction * (m->direction.w * f));
            addForce(pos, force);
        }
    }

    void addForce(const vec3 &pos, const vec3 &force)
    {
        atmCall(getParent(), addForce, atmArgs(pos, force));
    }
};
atmImplementEntity(GearParts);
atmExportClass(GearParts);


class dpPatch GearBase
    : public IEntity
    , public TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Transform> >
{
    typedef IEntity super;
    typedef TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Transform> > transform;

    stl::vector<EntityHandle> m_parts;
    EntityHandle m_linkage;
    float32 m_rot_angle;
    float32 m_rot_speed;
    float32 m_max_rot_speed;
    float32 m_rot_accel;
    float32 m_rot_decel;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerialize(m_linkage)
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
            atmECall(addRotateSpeed)
            atmECall(getLinkage)
            atmECall(setLinkage)
            atmECall(getRotateAngle)
            atmECall(setRotateAngle)
            atmECall(getRotateSpeed)
            atmECall(setRotateSpeed)
            atmECall(getMaxRotateSpeed)
            atmECall(setMaxRotateSpeed)
            atmECall(getRotateAccel)
            atmECall(setRotateAccel)
            atmECall(getRotateDecel)
            atmECall(setRotateDecel)
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

    void jsonize(stl::string &out) override
    {
        transform::jsonize(out);
    }

public:
    GearBase() : m_linkage(0), m_rot_angle(0.0f), m_rot_speed(0.0f), m_max_rot_speed(0.5f), m_rot_accel(0.00002f), m_rot_decel(0.99f)
    {
    }

    ~GearBase()
    {
        eachParts([&](EntityHandle h){ atmDeleteEntity(h); });
    }

    void initialize() override
    {
        transform::setAxis(vec3(0.0f, 0.0f, 1.0f));
    }

    void update(float32 dt) override
    {
        updateRotate(dt); // 子が参照するので asyncupdate() の中ではマズい
    }

    void asyncupdate(float32 dt) override
    {
    }

    void updateRotate(float32 dt)
    {
        m_rot_angle += m_rot_speed * dt;
        m_rot_speed *= m_rot_decel;
        transform::setRotate(m_rot_angle);
        transform::updateTransformMatrix();
    }

    void addForce(const vec3 &pos, const vec3 &force)
    {
        atmDbgAssertSyncLock();
        vec3 diff = pos-getPosition(); diff.z=0.0f;
        float32 dist = glm::length(diff);
        vec3 dir = vec3( glm::rotateZ(diff/dist, 90.0f) );
        float32 lf = glm::length(force);
        vec3 nf = force/lf;
        float32 d = glm::dot(dir, nf);
        float32 f = lf * m_rot_accel * dist * d;
        addRotateSpeed(f);
    }

    void addRotateSpeed(float32 v)      { setRotateSpeed(getRotateSpeed()+v); }
    void addParts(GearParts *v)         { m_parts.push_back(v->getHandle()); }
    EntityHandle getLinkage() const     { return m_linkage; }
    float32 getRotateAngle() const      { return m_rot_angle; }
    float32 getRotateSpeed() const      { return m_rot_speed; }
    float32 getMaxRotateSpeed() const   { return m_max_rot_speed; }
    float32 getRotateAccel() const      { return m_rot_accel; }
    float32 getRotateDecel() const      { return m_rot_decel; }
    void setLinkage(EntityHandle v)     { m_linkage=v; }
    void setRotateAngle(float32 v)      { m_rot_angle=v; }
    void setRotateSpeed(float32 v)      { m_rot_speed=glm::sign(v)*glm::min(glm::abs(v), m_max_rot_speed); }
    void setMaxRotateSpeed(float32 v)   { m_max_rot_speed=v; }
    void setRotateAccel(float32 v)      { m_rot_accel=v; }
    void setRotateDecel(float32 v)      { m_rot_decel=v; }
    template<class F> void eachParts(const F &f) { EachEntities(m_parts, f); }
};
atmExportClass(GearBase);


class dpPatch GearSmall : public GearBase
{
typedef GearBase super;
private:
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

    void initialize() override
    {
        setMaxRotateSpeed(1.5f);
        setRotateDecel(0.99f);
        setRotateAccel(0.0002f);

        const int32 div = 6;
        const vec4 dir_x(1.0f,0.0f,0.0f,0.0f);
        CollisionGroup cg = atmGetCollisionModule()->genGroup();
        for(int i=0; i<div; ++i) {
            vec3 dir = vec3(glm::rotateZ(dir_x, 360.0f/div*i));
            GearParts *e = (GearParts*)atmCreateEntityT(GearParts);
            e->setScale(vec3(1.5f, 0.25f, 0.75f));
            e->setParent(getHandle());
            e->setOrientation(dir);
            e->setCollisionGroup(cg);
            addParts(e);
        }
    }
};
atmImplementEntity(GearSmall, DF_Editor);
atmExportClass(GearSmall);


class dpPatch GearLarge : public GearBase
{
typedef GearBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    GearLarge()
    {
        wdmScope(
            wdmString path = wdmFormat("Level/GearLarge/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GearLarge()
    {
        wdmEraseNode(wdmFormat("Level/GearLarge/0x%p", this));
    }

    void initialize() override
    {
        setMaxRotateSpeed(0.6f);
        setRotateDecel(0.99f);
        setRotateAccel(0.00003f);

        const int32 div = 6;
        const vec4 dir_x(1.0f,0.0f,0.0f,0.0f);
        CollisionGroup cg = atmGetCollisionModule()->genGroup();
        for(int i=0; i<div; ++i) {
            vec3 dir = vec3(glm::rotateZ(dir_x, 360.0f/div*i));
            GearParts *e = (GearParts*)atmCreateEntityT(GearParts);
            e->setScale(vec3(2.5f, 0.4f, 1.0f));
            e->setParent(getHandle());
            e->setOrientation(dir);
            e->setCollisionGroup(cg);
            addParts(e);
        }
    }
};
atmImplementEntity(GearLarge, DF_Editor);
atmExportClass(GearLarge);

} // namespace atm
