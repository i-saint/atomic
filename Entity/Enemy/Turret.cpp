#include "stdafx.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"

namespace atm {


class LaserTurret : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(LaserTurret);
atmExportClass(LaserTurret);



} // namespace atm
