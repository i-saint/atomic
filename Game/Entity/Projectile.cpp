#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Enemy.h"
#include "Game/Entity/Routine.h"

namespace atm {

class dpPatch HomingMine : public Breakable<Entity_Orientation>
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


class dpPatch LaserMissile : public Breakable<Entity_Orientation>
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
