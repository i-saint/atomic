#include "stdafx.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/World.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_world(NULL)
{
    MessageRouter::initializeInstance();

    m_world = IST_NEW16(World)();
    m_world->initialize();
}

AtomicGame::~AtomicGame()
{
    IST_SAFE_DELETE(m_world);

    MessageRouter::finalizeInstance();
}


void AtomicGame::update(float32 dt)
{
    //MessageRouter *message_router = atomicGetMessageRouter(MR_SYSTEM);
    //message_router->unuseAll();

    m_world->update(dt);

    //message_router->route();
}


void AtomicGame::draw()
{
    atomicWaitForDrawComplete();
    // todo: フレームスキップ処理
    atomicKickDraw();
}

void AtomicGame::drawCallback()
{
    AtomicRenderer::getInstance()->beforeDraw();
    if(m_world) {
        m_world->draw();
    }
    AtomicRenderer::getInstance()->draw();

}



} // namespace atomic