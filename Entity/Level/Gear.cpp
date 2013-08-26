#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Level.h"

namespace atm {

class GearParts : public Unbreakable<Entity_Direction>
{
typedef Unbreakable<Entity_Direction>  super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )
    atmECallBlock(
        atmMethodBlock(
            atmECall(pulse)
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
        setPivot(vec3(-0.5f, 0.0f, -0.1f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender|CF_Receiver|CF_SPH_Sender);
        setModel(PSET_HOLLOW_CUBE);
    }


    void eventFluid(const FluidMessage *m) override
    {
        super::eventFluid(m);
        vec3 pos = (const vec3&)m->position;
        vec3 force = (const vec3&)m->velocity;
        pulse(pos, force);
    }

    void eventCollide(const CollideMessage *m) override
    {
        super::eventCollide(m);
        vec3 pos;
        if(atmQuery(m->from, getPosition, pos)) {
            float32 f = 1000.0f;
            if(EntityGetCategory(m->from)==ECA_Obstacle) {
                f = 10000.0f;
            }
            vec3 force = vec3(m->direction * (m->direction.w * f));
            pulse(pos, force);
        }
    }

    void pulse(const vec3 &pos, const vec3 &force)
    {
        atmCall(getParent(), pulse, atmArgs(pos, force));
    }
};
atmImplementEntity(GearParts);
atmExportClass(GearParts);



class GearBase
    : public IEntity
    , public TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Transform> >
    , public Attr_Spin
{
    typedef IEntity super;
    typedef TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Transform> > transform;
    typedef Attr_Spin spin;

    stl::vector<EntityHandle> m_parts;

    istSerializeBlock(
        istSerializeBase(super)
        istSerializeBase(transform)
        istSerializeBase(spin)
        istSerialize(m_parts)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
        atmECallSuper(transform)
        atmECallSuper(spin)
    )

    wdmScope(
    void addDebugNodes(const wdmString &path)
    {
        transform::addDebugNodes(path);
        spin::addDebugNodes(path);
    }
    )

    void jsonize(stl::string &out) override
    {
        transform::jsonize(out);
        spin::jsonize(out);
    }

public:
    GearBase()
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
        if(isParentDead()) {
            atmDeleteEntity(getHandle());
            return;
        }
        spin::updateSpin(dt, getPositionAbs()); // 子が参照するので asyncupdate() の中ではマズい
        transform::setRotate(getSpinAngle());
        transform::updateTransformMatrix();
    }

    void asyncupdate(float32 dt) override
    {
    }

    void addParts(GearParts *v)     { m_parts.push_back(v->getHandle()); }
    template<class F> void eachParts(const F &f) { EachEntities(m_parts, f); }
};
atmExportClass(GearBase);




class GearSmall : public GearBase
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
        setSpinMaxSpeed(4.0f);
        setSpinDecel(0.99f);
        setSpinResist(0.004f);

        const int32 div = 5;
        const vec4 dir_x(1.0f,0.0f,0.0f,0.0f);
        CollisionGroup cg = atmGetCollisionModule()->genGroup();
        for(int i=0; i<div; ++i) {
            vec3 dir = vec3(glm::rotateZ(dir_x, 360.0f/div*i));
            GearParts *e = (GearParts*)atmCreateEntityT(GearParts);
            e->setScale(vec3(0.3f, 0.1f, 0.15f));
            e->setParent(getHandle());
            e->setDirection(dir);
            e->setCollisionGroup(cg);
            addParts(e);
        }
    }
};
atmImplementEntity(GearSmall, DF_Editor);
atmExportClass(GearSmall);


class GearMedium : public GearBase
{
typedef GearBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    GearMedium()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GearMedium/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GearMedium()
    {
        wdmEraseNode(wdmFormat("Level/GearMedium/0x%p", this));
    }

    void initialize() override
    {
        setSpinMaxSpeed(1.5f);
        setSpinDecel(0.99f);
        setSpinResist(0.0003f);

        const int32 div = 6;
        const vec4 dir_x(1.0f,0.0f,0.0f,0.0f);
        CollisionGroup cg = atmGetCollisionModule()->genGroup();
        for(int i=0; i<div; ++i) {
            vec3 dir = vec3(glm::rotateZ(dir_x, 360.0f/div*i));
            GearParts *e = (GearParts*)atmCreateEntityT(GearParts);
            e->setScale(vec3(0.5f, 0.1f, 0.25f));
            e->setParent(getHandle());
            e->setDirection(dir);
            e->setCollisionGroup(cg);
            addParts(e);
        }
    }
};
atmImplementEntity(GearMedium, DF_Editor);
atmExportClass(GearMedium);


class GearLarge : public GearBase
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
        setSpinMaxSpeed(0.6f);
        setSpinDecel(0.99f);
        setSpinResist(0.00004f);

        const int32 div = 6;
        const vec4 dir_x(1.0f,0.0f,0.0f,0.0f);
        CollisionGroup cg = atmGetCollisionModule()->genGroup();
        for(int i=0; i<div; ++i) {
            vec3 dir = vec3(glm::rotateZ(dir_x, 360.0f/div*i));
            GearParts *e = (GearParts*)atmCreateEntityT(GearParts);
            e->setScale(vec3(1.0f, 0.15f, 0.3f));
            e->setParent(getHandle());
            e->setDirection(dir);
            e->setCollisionGroup(cg);
            addParts(e);
        }
    }
};
atmImplementEntity(GearLarge, DF_Editor);
atmExportClass(GearLarge);



class GearExLarge : public GearBase
{
typedef GearBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    GearExLarge()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GearExLarge/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GearExLarge()
    {
        wdmEraseNode(wdmFormat("Level/GearExLarge/0x%p", this));
    }

    void initialize() override
    {
        setSpinMaxSpeed(0.45f);
        setSpinDecel(0.99f);
        setSpinResist(0.000025f);

        const int32 div = 8;
        const vec4 dir_x(1.0f,0.0f,0.0f,0.0f);
        CollisionGroup cg = atmGetCollisionModule()->genGroup();
        for(int i=0; i<div; ++i) {
            vec3 dir = vec3(glm::rotateZ(dir_x, 360.0f/div*i));
            GearParts *e = (GearParts*)atmCreateEntityT(GearParts);
            e->setScale(vec3(1.25f, 0.2f, 0.3f));
            e->setParent(getHandle());
            e->setDirection(dir);
            e->setCollisionGroup(cg);
            addParts(e);
        }
    }
};
atmImplementEntity(GearExLarge, DF_Editor);
atmExportClass(GearExLarge);

} // namespace atm
