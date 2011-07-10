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
    m_worlds[0] = IST_NEW16(World)();
    m_worlds[0]->initialize();
}

AtomicGame::~AtomicGame()
{
    for(uint32 i=0; i<MAX_WORLDS; ++i) {
        IST_DELETE(m_worlds[i]);
    }

    World::finalizeInterframe();
    MessageRouter::finalizeInstance();
    AtomicRenderer::finalizeInstance();
}


void AtomicGame::update()
{
    uint32 next_world = (m_current_world+1) % MAX_WORLDS;
    World *w = m_worlds[m_current_world];
    World *n = m_worlds[next_world];
    if(!n) {
        n = IST_NEW16(World)();
        //n = w;
        m_worlds[next_world] = n;
    }

    //MessageRouter *message_router = atomicGetMessageRouter(MR_SYSTEM);
    //message_router->unuseAll();

    w->setNext(n);
    w->update();
    m_draw_target = w;

    //message_router->route();

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