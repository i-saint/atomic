#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
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

void World::InitializeInterframe()
{
    if(!s_interframe) {
        s_interframe = EA_ALIGNED_NEW(Interframe, 16) Interframe();
    }
    FractionSet::InitializeInterframe();
}

void World::FinalizeInterframe()
{
    FractionSet::FinalizeInterframe();
    EA_DELETE(s_interframe);
}


World::World(World* prev) : m_prev(prev), m_next(NULL)
{
    m_rand.initialize(0);
    m_fraction_set = EA_ALIGNED_NEW(FractionSet, 16) FractionSet(NULL, NULL);

    m_camera.setPosition(XMVectorSet(100.0f, 100.0f, 500.0f, 0.0f));
    m_camera.setZNear(1.0f);
    m_camera.setZFar(1000.0f);
}

World::~World()
{
    EA_DELETE(m_fraction_set);

    if(m_prev) { m_prev->m_next = m_next; }
    if(m_next) { m_next->m_prev = m_prev; }
}

void World::update()
{
    getInterframe()->setCurrentWorld(this);

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
    {
        float light_pos[] = {10000.0f, 10000.0f, 10000.0f, 0.0f};
        float ambient[] = {0.2f, 0.2f, 0.2f, 1.0f};
        float diffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
        float specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
        //glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
        //glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
    }
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    m_camera.setPosition(XMVector3Transform(m_camera.getPosition(), XMMatrixRotationY(XMConvertToRadians(0.1f))));
    m_camera.bind();

    m_fraction_set->draw();

    // todo: KickDrawCommand()
}


} // namespace atomic
