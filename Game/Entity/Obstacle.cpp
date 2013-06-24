#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"

namespace atm {


class dpPatch PointLightEntity : public IEntity
{

};

class dpPatch DirectionalLightEntity : public IEntity
{

};


class dpPatch GroundBlock : public EntityTemplate<Entity_Orientation>
{
typedef EntityTemplate<Entity_Orientation>  super;
private:
    EntityHandle    m_parent;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_parent)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
        atmMethodBlock(
            atmECall(setParent)
            atmECall(getParent)
        )
    )

public:
    GroundBlock() : m_parent(0)
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GroundBlock[0x%p]", this);
        super::addDebugNodes(path);
        )
    }

    ~GroundBlock()
    {
        wdmEraseNode(wdmFormat("Level/GroundBlock[0x%p]", this));
    }

    void            setParent(EntityHandle v)    { m_parent=v; }
    EntityHandle    getParent() const            { return m_parent; }

    virtual void initialize()
    {
        setPivot(vec3(-0.2f, 0.0f, -0.1f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender|CF_SPH_Sender);

        setModel(PSET_CUBE_MEDIUM);
        setGlowColor(vec4(0.4f));
        setDiffuseColor(vec4(vec3(0.5f), 50.0f));
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
        collision::updateCollisionByParticleSet(getModel(), getTransform());
        bloodstain::updateBloodstain(dt);
    }

    virtual void draw()
    {
        PSetInstance inst;
        inst.diffuse = getDiffuseColor();
        inst.glow = getGlowColor();
        inst.flash = vec4();
        inst.elapsed = 1000.0f;
        inst.appear_radius = 10000.0f;
        inst.translate = getTransform();
        atmGetSPHPass()->addPSetInstance(getModel(), inst);
        atmGetBloodStainPass()->addBloodstainParticles(getTransform(), getBloodStainParticles(), getNumBloodstainParticles());
    }

    virtual void eventFluid(const FluidMessage *m)
    {
        addBloodstain(getInverseTransform(), (vec4&)m->position);
    }
};
atmImplementEntity(GroundBlock);
atmExportClass(atm::GroundBlock);

} // namespace atm
