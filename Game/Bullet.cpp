#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/Message.h"
#include "Game/World.h"
#include "Game/Bullet.h"
#include "Game/Bullet_Octahedron.h"


namespace atomic {



BulletSet::BulletSet()
{
}

BulletSet::~BulletSet()
{
    uint32 num_subsets = m_subsets.size();
    for(uint32 i=0; i<num_subsets; ++i) {
        IST_SAFE_DELETE(m_subsets[i]);
    }
    m_subsets.clear();
}

} // namespace atomic
