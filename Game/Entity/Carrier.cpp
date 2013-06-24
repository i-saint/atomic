#include "stdafx.h"
#include "Game/Entity/EntityCommon.h"
#include "Game/Entity/Enemy.h"
#include "Game/Entity/Routine.h"

namespace atm {

class dpPatch HatchBase : public Breakable<Entity_Orientation>
{
};

class dpPatch HatchSmall : public HatchBase
{
};

class dpPatch HatchLarge : public HatchBase
{
};



class dpPatch CarrierBase  : public IEntity
{
};

class dpPatch CarrierSmall  : public IEntity
{
};

class dpPatch CarrierLarge  : public IEntity
{
};

} // namespace atm
