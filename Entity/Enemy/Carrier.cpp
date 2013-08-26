#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"

namespace atm {


class CarrierBase  : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
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
