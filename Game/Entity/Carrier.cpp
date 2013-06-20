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
typedef HatchBase this_t;
};

class dpPatch HatchSmall : public HatchBase
{
typedef HatchSmall this_t;
};

class dpPatch HatchLarge : public HatchBase
{
typedef HatchLarge this_t;
};



class dpPatch CarrierBase  : public IEntity
{
typedef CarrierBase this_t;
};

class dpPatch CarrierSmall  : public IEntity
{
typedef CarrierSmall this_t;
};

class dpPatch CarrierLarge  : public IEntity
{
typedef CarrierLarge this_t;
};

} // namespace atm
