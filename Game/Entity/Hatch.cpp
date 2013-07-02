#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Enemy.h"
#include "Game/Entity/Routine.h"

namespace atm {


class dpPatch HatchBase : public Breakable<Entity_Orientation>
{
typedef Breakable<Entity_Orientation> super;
private:
    int32 m_frame;

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_frame)
    )

public:
    atmECallBlock(
        atmECallSuper(super)
    )

public:
    HatchBase() : m_frame(0)
    {
        wdmScope(
        wdmString path = wdmFormat("Enemy/Hatch/0x%p", this);
        super::addDebugNodes(path);
        )
    }

    ~HatchBase()
    {
        wdmEraseNode(wdmFormat("Level/Hatch/0x%p", this));
    }


    void initialize() override
    {
    }

    void update(float32 dt) override
    {
        super::update(dt);
        ++m_frame;
    }

    void asyncupdate(float32 dt) override
    {
        super::asyncupdate(dt);
        transform::updateTransformMatrix();
        bloodstain::updateBloodstain(dt);
    }
};
atmExportClass(HatchBase);


class dpPatch HatchSmall : public HatchBase
{
typedef HatchBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(HatchSmall);
atmExportClass(HatchSmall);


class dpPatch HatchLarge : public HatchBase
{
typedef HatchBase super;
private:
public:
};
atmImplementEntity(HatchLarge);
atmExportClass(HatchLarge);


} // namespace atm
