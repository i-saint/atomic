#include "stdafx.h"
#include "../types.h"
#include "Message.h"
#include "World.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "../Graphics/GraphicResourceManager.h"
#include "../Graphics/Renderer.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_current_world(0)
, m_draw_target(NULL)
{
    AtomicRenderer::initializeInstance();
    MessageRouter::initializeInstance();
    World::initializeInterframe();

    m_worlds.resize(MAX_WORLDS);
    m_worlds[0] = AT_ALIGNED_NEW(World, 16)();
    m_worlds[0]->initialize(NULL);
}

AtomicGame::~AtomicGame()
{
    for(uint32 i=0; i<MAX_WORLDS; ++i) {
        AT_DELETE(m_worlds[i]);
    }

    World::finalizeInterframe();
    MessageRouter::finalizeInstance();
    AtomicRenderer::finalizeInstance();
}


void AtomicGame::update()
{
    MessageRouter *message_router = atomicGetMessageRouter(MR_SYSTEM);
    message_router->unuseAll();

    World *w = m_worlds[m_current_world];
    w->update();
    m_draw_target = w;

    message_router->route();


    uint32 next_world = (m_current_world+1) % MAX_WORLDS;
    World *n = m_worlds[next_world];
    if(!n) {
        //n = AT_ALIGNED_NEW(World, 16)();
        n = w;
    }
    n->initialize(w);
    m_worlds[next_world] = n;
    m_current_world = next_world;
}


void AtomicGame::draw()
{
    atomicWaitForDrawComplete();
    AtomicRenderer::getInstance()->beforeDraw();
    if(m_draw_target) {
        m_draw_target->draw();
    }
    atomicKickDraw();
}



} // namespace atomic