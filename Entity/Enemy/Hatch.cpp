#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"

namespace atm {


class HatchBase : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
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


class HatchSmall : public HatchBase
{
typedef HatchBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(HatchSmall, DF_Editor, 4.0f);
atmExportClass(HatchSmall);


class HatchLarge : public HatchBase
{
typedef HatchBase super;
private:
public:
};
atmImplementEntity(HatchLarge, DF_Editor, 8.0f);
atmExportClass(HatchLarge);


} // namespace atm
