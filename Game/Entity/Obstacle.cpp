#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Level.h"

namespace atm {


class GroundBlock : public Unbreakable<Entity_Orientation>
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
        setPivot(vec3(-0.2f, 0.0f, -0.1f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_Sender|CF_SPH_Sender);

        setModel(PSET_CUBE_MEDIUM);
    }
};
atmImplementEntity(GroundBlock);
atmExportClass(GroundBlock);



class FluidFilter : public Unbreakable<Entity_Orientation>
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
    FluidFilter()
    {
        wdmScope(
        wdmString path = wdmFormat("Level/GroundBlock/0x%p", this);
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
        setPivot(vec3(-0.2f, 0.0f, -0.1f));

        initializeCollision(getHandle());
        setCollisionShape(CS_Box);
        setCollisionFlags(CF_SPH_Sender);

        setModel(PSET_CUBE_MEDIUM);
    }

    void draw() override
    {
        // todo
    }
};
atmImplementEntity(FluidFilter);
atmExportClass(FluidFilter);


} // namespace atm
