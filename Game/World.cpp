#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "AtomicApplication.h"
#include "Message.h"
#include "Fraction.h"
#include "World.h"

using namespace ist::graphics;


namespace atomic
{

World::Interframe::Interframe()
: m_current_world(NULL)
{
}


World::Interframe *World::s_interframe;

void World::initializeInterframe()
{
    if(!s_interframe) {
        s_interframe = AT_ALIGNED_NEW(Interframe, 16)();
    }
    FractionSet::InitializeInterframe();
}

void World::finalizeInterframe()
{
    FractionSet::FinalizeInterframe();
    AT_DELETE(s_interframe);
}


World::World()
: m_prev(NULL)
, m_fraction_set(NULL)
{
    m_fraction_set = AT_ALIGNED_NEW(FractionSet, 16)();

    m_rand.initialize(0);
    m_camera.setPosition(XMVectorSet(100.0f, 100.0f, 500.0f, 0.0f));
    m_camera.setZNear(1.0f);
    m_camera.setZFar(1000.0f);
}

World::~World()
{
    AT_DELETE(m_fraction_set);
}

void World::initialize( World* prev, FrameAllocator& alloc )
{
    m_prev = prev;
    if(prev) {
        m_rand = prev->m_rand;
        m_camera = prev->m_camera;
    }
    m_fraction_set->initialize(prev ? prev->m_fraction_set : NULL, alloc);
}

void World::update()
{
    getInterframe()->setCurrentWorld(this);

    m_camera.setPosition(XMVector3Transform(m_camera.getPosition(), XMMatrixRotationY(XMConvertToRadians(0.1f))));
    m_camera.setAspect(GetWindowAspectRatio());

    m_fraction_set->update();
}

void World::sync()
{
    m_fraction_set->sync();
}

void World::flushMessage()
{
    m_fraction_set->flushMessage();
}

void World::processMessage()
{
    m_fraction_set->processMessage();

    size_t required_memory = 0;
    required_memory += m_fraction_set->getRequiredMemoryOnNextFrame();
}

void World::draw()
{
    m_fraction_set->draw();

    // todo: KickDrawCommand()
}


} // namespace atomic
