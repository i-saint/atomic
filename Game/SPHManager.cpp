#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"

namespace atomic {




SPHManager::SPHManager()
{
}

SPHManager::~SPHManager()
{
}


void SPHManager::initialize()
{
    {
        sphForcePointGravity h_sg;
        h_sg.position = make_float4(0.0f);
        h_sg.inner_radus = 0.25f;
        h_sg.range_radus = 5.12f;
        h_sg.strength = 1.0f;
        for(uint32 i=0; i<_countof(m_sgravity); ++i) {
            m_sgravity[i] = h_sg;
        }

    }
}


void SPHManager::updateBegin( float32 dt )
{
    m_rigids.clear();
}

void SPHManager::update(float32 dt)
{
}

void SPHManager::asyncupdate(float32 dt)
{
    vec2 move = atomicGetInputs()->getMove()*0.01f;
    m_sgravity[0].position.x += move.x;
    m_sgravity[0].position.y += move.y;

    SPHUpdateGravity(m_sgravity);
    SPHUpdateRigids(m_rigids);
    SPHUpdateFluid();
}

void SPHManager::addRigid(CB_RID cid, EntityHandle h, const mat4& m)
{
    sphRigidInstance tmp;
    tmp.classid = cid;
    tmp.handle = h;
    tmp.transform = m;
    m_rigids.push_back( tmp );
}

void SPHManager::draw() const
{
}


} // namespace atomic
