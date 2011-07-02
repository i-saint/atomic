#include "stdafx.h"
#include "../types.h"
#include "World.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "../Graphics/GraphicResourceManager.h"
#include "../Graphics/Renderer.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_world(NULL)
{
    AtomicRenderer::initializeInstance();
    World::initializeInterframe();
    m_world = AT_ALIGNED_NEW(World, 16) World(NULL);
}

AtomicGame::~AtomicGame()
{
    AT_DELETE(m_world);
    World::finalizeInterframe();
    AtomicRenderer::finalizeInstance();
}


void AtomicGame::update()
{
    m_world->update();
}


void AtomicGame::draw()
{
    WaitForDrawComplete();
    AtomicRenderer::getInstance()->beforeDraw();
    m_world->draw();
    KickDraw();
}



} // namespace atomic