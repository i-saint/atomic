#include "stdafx.h"
#include "../types.h"
#include "World.h"
#include "AtomicGame.h"

namespace atomic {


AtomicGame::AtomicGame()
{
    World::InitializeInterframe();
    m_world = EA_ALIGNED_NEW(World, 16) World(NULL);
}

AtomicGame::~AtomicGame()
{
    EA_DELETE(m_world);
    World::FinalizeInterframe();
}

void AtomicGame::update()
{
    m_world->update();
}


void AtomicGame::draw()
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glShadeModel(GL_SMOOTH);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);

    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glEnable(GL_DEPTH_TEST);

    m_world->draw();

    glSwapBuffers();
}

} // namespace atomic