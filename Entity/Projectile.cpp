#include "stdafx.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"

namespace atm {

class HomingMine : public Breakable<Entity_Orientation>
{
typedef Breakable<Entity_Orientation> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(HomingMine);
atmExportClass(HomingMine);


class LaserMissile : public Breakable<Entity_Orientation>
{
typedef Breakable<Entity_Orientation> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(LaserMissile);
atmExportClass(LaserMissile);

} // namespace atm
