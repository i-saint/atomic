#include "stdafx.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/World.h"
#include "Util.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_world(NULL)
, m_input_server(NULL)
{
    MessageRouter::initializeInstance();

    m_input_server = IST_NEW(InputServerLocal)();

    m_world = IST_NEW(World)();
    m_world->initialize();
}

AtomicGame::~AtomicGame()
{
    if(m_input_server->getClassID()==IInputServer::IS_LOCAL) {
        char path[128];
        char date[128];
        CreateDateString(date, _countof(date));
        sprintf_s(path, _countof(path), "%s.replay", date);
        for(size_t i=0; i<_countof(path); ++i) { if(path[i]=='/' || path[i]==':') { path[i]='-'; } }
        static_cast<InputServerLocal*>(m_input_server)->writeToFile(path);
    }

    IST_SAFE_DELETE(m_world);
    IST_SAFE_DELETE(m_input_server);

    MessageRouter::finalizeInstance();
}

bool AtomicGame::readReplayFromFile(const char *path)
{
    InputServerReplay *ris = IST_NEW(InputServerReplay)();
    if(ris->readFromFile(path)) {
        IST_DELETE(m_input_server);
        m_input_server = ris;
        return true;
    }

    IST_DELETE(ris);
    return false;
}


void AtomicGame::update(float32 dt)
{
    m_input_server->update(*atomicGetSystemInputs());
    m_world->update(dt);
}

void AtomicGame::asyncupdateBegin(float32 dt)
{
    m_world->asyncupdateBegin(dt);
}

void AtomicGame::asyncupdateEnd()
{
    m_world->asyncupdateEnd();
}


void AtomicGame::draw()
{
    // todo: フレームスキップ処理
    atomicKickDraw();
    atomicWaitForDrawComplete();
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