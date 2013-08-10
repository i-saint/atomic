#include "stdafx.h"
#include "Entity/EntityCommon.h"
#include "Entity/Level.h"

namespace atm {


class GroundBlock : public Unbreakable<Entity_Direction>
{
typedef Unbreakable<Entity_Direction>  super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
    )
    atmJsonizeBlock(
        atmJsonizeSuper(super)
    )

public:
    GroundBlock()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GroundBlock/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~GroundBlock()
    {
        wdmEraseNode(wdmFormat("Level/GroundBlock/0x%p", this));
    }

    void initialize() override
    {
        super::initialize();
        initializeCollision(getHandle());
        setScale(vec3(0.5, 0.5f, 0.2f));
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender|CF_SPH_Sender);
        setModel(PSET_HOLLOW_CUBE);
    }
};
atmImplementEntity(GroundBlock, DF_Editor, 0.0f);
atmExportClass(GroundBlock);



class FluidFilter : public Unbreakable<Entity_Direction>
{
typedef Unbreakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
    )

public:
    FluidFilter()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/FluidFilter/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~FluidFilter()
    {
        wdmEraseNode(wdmFormat("Level/FluidFilter/0x%p", this));
    }

    void initialize() override
    {
        super::initialize();
        initializeCollision(getHandle());
        setScale(vec3(0.5, 0.5f, 0.2f));
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_SPH_Sender);
        setModel(PSET_HOLLOW_CUBE);
    }

    void draw() override
    {
        PSetInstance inst;
        inst.diffuse = getDiffuseColor();
        inst.glow = getGlowColor();
        inst.flash = vec4();
        inst.elapsed = pasttime::getPastTime();
        inst.appear_radius = inst.elapsed * 0.004f;
        inst.transform = transform::getTransformMatrix();
        inst.rotate = transform::computeRotationMatrix();
        uint32 num = 0;
        vec3 size;
        if(atmQuery(this, getScale, size)) {
            num = uint32(size.x*size.y*size.z * 10000.0f);
        }
        atmGetBarrierPass()->addParticles(getModel(), inst, num);
    }
};
atmImplementEntity(FluidFilter, DF_Editor, 0.0f);
atmExportClass(FluidFilter);


class dpPatch RigidFilter : public Unbreakable<Entity_Direction>
{
typedef Unbreakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
    )

public:
    RigidFilter()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/RigidFilter/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~RigidFilter()
    {
        wdmEraseNode(wdmFormat("Level/RigidFilter/0x%p", this));
    }

    void initialize() override
    {
        super::initialize();

        initializeCollision(getHandle());
        setScale(vec3(0.5, 0.5f, 0.2f));
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender);
        setModel(PSET_UNIT_CUBE);
    }

    void draw() override
    {
        PSetInstance inst;
        inst.diffuse = getDiffuseColor();
        inst.glow = getGlowColor();
        inst.flash = vec4();
        inst.elapsed = pasttime::getPastTime();
        inst.appear_radius = inst.elapsed * 0.004f;
        inst.transform = transform::getTransformMatrix();
        inst.rotate = transform::computeRotationMatrix();
        uint32 num = 0;
        vec3 size;
        if(atmQuery(this, getScale, size)) {
            num = uint32((size.x*1.2f)*(size.y*1.2f)*size.z * 1500.0f);
        }
        atmGetFluidPass()->addParticlesSolid(getModel(), inst, num);
    }
};
atmImplementEntity(RigidFilter, DF_Editor, 0.0f);
atmExportClass(RigidFilter);

} // namespace atm
