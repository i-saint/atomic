#include "stdafx.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Graphics/GraphicResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/World.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_current(NULL)
, m_prev(NULL)
, m_world_index(0)
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
    uint32 next_world = (m_world_index+1) % MAX_WORLDS;
    World *w = m_worlds[m_world_index];
    World *n = m_worlds[next_world];
    if(!n) {
        n = IST_NEW16(World)();
        //n->initialize();
        //n = w;
        m_worlds[next_world] = n;
    }

    //MessageRouter *message_router = atomicGetMessageRouter(MR_SYSTEM);
    //message_router->unuseAll();

    w->setNext(n);
    w->update();
    m_draw_target = w;

    //message_router->route();

    m_current = n;
    m_prev = w;
    m_world_index = next_world;
}


void AtomicGame::draw()
{
    atomicWaitForDrawComplete();
    // todo: フレームスキップ処理
    m_draw_target = m_prev;
    atomicKickDraw();
}

void AtomicGame::drawCallback()
{
    AtomicRenderer::getInstance()->beforeDraw();
    if(m_draw_target) {
        m_draw_target->draw();
    }
    AtomicRenderer::getInstance()->draw();

}



} // namespace atomic