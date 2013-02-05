#include "stdafx.h"
#include "features.h"
#include "types.h"
#include "Input.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "World.h"
#include "Network/LevelEditorServer.h"

namespace atomic {

namespace  {
    const char magic_string[8] = "atomic\x00";
}

RepHeader::RepHeader()
{
    istMemset(this, 0, sizeof(*this));
    istMemcpy(magic, magic_string, sizeof(magic));
    version = atomic_replay_version;
}

bool RepHeader::isValid()
{
    if( istMemcmp(magic, magic_string, sizeof(magic))==0 && version==atomic_replay_version)
    {
        return true;
    }
    return false;
}

RepPlayer::RepPlayer() { istMemset(this, 0, sizeof(*this)); }





} // namespace atomic
