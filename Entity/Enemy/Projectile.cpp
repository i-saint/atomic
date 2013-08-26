#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"

namespace atm {

class HomingMine : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(HomingMine);
atmExportClass(HomingMine);


class LaserMissile : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(LaserMissile);
atmExportClass(LaserMissile);

} // namespace atm
