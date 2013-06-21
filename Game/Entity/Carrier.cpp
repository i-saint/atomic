#include "stdafx.h"
#include "types.h"
#include "Util.h"
#include "Sound/AtomicSound.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "Game/Collision.h"
#include "Game/Message.h"
#include "Routine.h"
#include "Enemy.h"

namespace atm {

class dpPatch HatchBase
    : public IEntity
    , public Attr_MessageHandler
    , public Attr_ParticleSet
    , public Attr_Collision
    , public Attr_Bloodstain
    , public TAttr_TransformMatrixI<Attr_Orientation>
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
