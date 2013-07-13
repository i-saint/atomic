#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Enemy.h"
#include "Game/Entity/Routine.h"

namespace atm {


class CarrierBase  : public Breakable<Entity_Orientation>
{
typedef Breakable<Entity_Orientation> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmExportClass(CarrierBase);


class CarrierSmall  : public CarrierBase
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


class CarrierLarge  : public CarrierBase
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
