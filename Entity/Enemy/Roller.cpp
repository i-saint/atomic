#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"
#include "Entity/Level.h"

namespace atm {

class BreakableParts : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(BreakableParts);
atmExportClass(BreakableParts);


class BreakableCore : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(BreakableCore);
atmExportClass(BreakableCore);




class RollerParts : public Unbreakable<Entity_Direction>
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
    RollerParts()
    {
        wdmScope(
            wdmString path = wdmFormat("Level/RollerParts/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~RollerParts()
    {
        wdmEraseNode(wdmFormat("Level/RollerParts/0x%p", this));
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
atmImplementEntity(RollerParts);
atmExportClass(RollerParts);


class RollerBase
    : public IEntity
    , public TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Direction> >
{
typedef IEntity super;
typedef TAttr_TransformMatrixI< TAttr_HaveParent<Attr_Direction> > transform;
private:
    vec3 m_grounding;
    vec3 m_center_to_grounding;
    vec3 m_gravity;
    float32 m_roll_angle;
    float32 m_roll_speed;
    bool m_grounded;
    stl::vector<EntityHandle> m_parts;

    istSerializeBlock(
        istSerializeBase(transform)
        istSerialize(m_grounding)
        istSerialize(m_center_to_grounding)
        istSerialize(m_gravity)
        istSerialize(m_roll_angle)
        istSerialize(m_roll_speed)
        istSerialize(m_grounded)
        istSerialize(m_parts)
    )
    atmECallBlock(
        atmECallSuper(transform)
    )

public:
    RollerBase()
        : m_grounding(), m_center_to_grounding(1.0f,0.0f,0.0f), m_gravity(0.0f,-0.001f,0.0f)
        , m_roll_angle(0.0f), m_roll_speed(0.0f), m_grounded(false)
    {
    }

    ~RollerBase()
    {
        eachParts([&](EntityHandle h){ atmDeleteEntity(h); });
    }

    void addParts(RollerParts *v)     { m_parts.push_back(v->getHandle()); }
    template<class F> void eachParts(const F &f) { EachEntities(m_parts, f); }

    void update(float32 dt) override
    {
        if(isParentDead()) {
            atmDeleteEntity(getHandle());
            return;
        }
        transform::updateTransformMatrix();
    }
};
atmExportClass(RollerBase);


class TriRollerSmall : public RollerBase
{
typedef RollerBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(TriRollerSmall);
atmExportClass(TriRollerSmall);


class TriRollerLarge : public RollerBase
{
typedef RollerBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(TriRollerLarge);
atmExportClass(TriRollerLarge);


class PentaRoller : public RollerBase
{
typedef RollerBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(PentaRoller);
atmExportClass(PentaRoller);




class Shell : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(Shell);
atmExportClass(Shell);


class Zab : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(Zab);
atmExportClass(Zab);


class SmallNucleus : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(SmallNucleus);
atmExportClass(SmallNucleus);



} // namespace atm
