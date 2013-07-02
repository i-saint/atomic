#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Enemy.h"
#include "Game/Entity/Routine.h"

namespace atm {


class dpPatch CarrierBase  : public Breakable<Entity_Orientation>
{
typedef Breakable<Entity_Orientation> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmExportClass(CarrierBase);


class dpPatch CarrierSmall  : public CarrierBase
{
typedef CarrierBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(CarrierSmall);
atmExportClass(CarrierSmall);


class dpPatch CarrierLarge  : public CarrierBase
{
typedef CarrierBase super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(CarrierLarge);
atmExportClass(CarrierLarge);


} // namespace atm
