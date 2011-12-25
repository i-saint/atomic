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
        sphForcePointGravity pg;
        pg.position = make_float4(0.0f);
        pg.inner_radus = 0.25f;
        pg.range_radus = 5.12f;
        pg.strength = 1.0f;
        m_pgravity.resize(1);
        m_pgravity[0] = pg;
    }
}


void SPHManager::updateBegin( float32 dt )
{
    m_rigids.clear();
    m_spheres.clear();
    m_boxes.clear();
}

void SPHManager::update(float32 dt)
{
}

void SPHManager::asyncupdate(float32 dt)
{
    vec2 move = atomicGetInputs()->getMove()*0.01f;
    m_pgravity[0].position.x += move.x;
    m_pgravity[0].position.y += move.y;

    SPHUpdateGravity(m_pgravity);
    SPHUpdateRigids(m_rigids, m_spheres, m_boxes);
    SPHUpdateFluid();
}

void SPHManager::addRigidInstance(CB_RID cid, EntityHandle h, const mat4 &m)
{
    sphRigidInstance tmp;
    tmp.classid = cid;
    tmp.handle = h;
    tmp.transform = m;
    m_rigids.push_back( tmp );
}

void SPHManager::addRigidSphere(CB_RID cid, EntityHandle h, const mat4 &m, const sphRigidSphere &s)
{
    addRigidInstance(cid, h, m);
    m_spheres.push_back(s);
}

void SPHManager::addRigidBox(CB_RID cid, EntityHandle h, const mat4 &m, const sphRigidBox &s)
{
    addRigidInstance(cid, h, m);
    m_boxes.push_back(s);
}

void SPHManager::draw() const
{
}


} // namespace atomic
