#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Level.h"

namespace atm {


class dpPatch GroundBlock : public Unbreakable<Entity_Orientation>
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

    virtual void initialize() override
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

} // namespace atm
