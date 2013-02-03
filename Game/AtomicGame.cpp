#include "stdafx.h"
#include "types.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "Graphics/AtomicRenderingSystem.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/World.h"
#include "Util.h"
#include "Game/Entity.h"
#include "Game/EntityQuery.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_world(NULL)
, m_input_server(NULL)
{
#ifdef atomic_enable_sync_lock
    m_sync_lock = false;
#endif // atomic_enable_sync_lock
    MessageRouter::initializeInstance();

    m_input_server = istNew(InputServerLocal)();

    m_world = istNew(World)();
    m_world->initialize();

    // 今回は固定値で初期化
    m_rand.initialize(0);
}

AtomicGame::~AtomicGame()
{
    if(m_input_server->getClassID()==IInputServer::IS_LOCAL && atomicGetConfig()->output_replay)
    {
        char path[128];
        char date[128];
        CreateDateString(date, _countof(date));
        istsprintf(path, "%s.replay", date);
        for(size_t i=0; i<_countof(path); ++i) { if(path[i]=='/' || path[i]==':') { path[i]='-'; } }
        static_cast<InputServerLocal*>(m_input_server)->writeToFile(path);
    }

    istSafeDelete(m_world);
    istSafeDelete(m_input_server);

    MessageRouter::finalizeInstance();
}

bool AtomicGame::readReplayFromFile(const char *path)
{
    InputServerReplay *ris = istNew(InputServerReplay)();
    if(ris->readFromFile(path)) {
        istDelete(m_input_server);
        m_input_server = ris;
        return true;
    }

    istDelete(ris);
    return false;
}


void AtomicGame::frameBegin()
{
    if(atomicGetConfig()->pause) { return; }

    m_world->frameBegin();
}

void AtomicGame::update(float32 dt)
{
    if(!atomicDbgDebugMenuIsActive()) { m_input_server->update(*atomicGetSystemInputs()); }
    LevelEditorServer::getInstance()->handleCommands(std::bind(&AtomicGame::handleLevelEditorCommands, this, std::placeholders::_1) );
    LevelEditorServer::getInstance()->handleQueries(std::bind(&AtomicGame::handleLevelEditorQueries, this, std::placeholders::_1) );
    if(!atomicGetConfig()->pause) {
        m_world->update(1.0f);
    }
}

void AtomicGame::asyncupdateBegin(float32 dt)
{
    if(atomicGetConfig()->pause) { return; }

    atomicDbgLockSyncMethods();
    m_world->asyncupdateBegin(dt);
}

void AtomicGame::asyncupdateEnd()
{
    if(atomicGetConfig()->pause) { return; }

    m_world->asyncupdateEnd();
    atomicDbgUnlockSyncMethods();
}


void AtomicGame::draw()
{
    // todo: フレームスキップ処理

    if(m_input_server->getClassID()==IInputServer::IS_REPLAY) {
        static uint32 f;
        const InputState *is = atomicGetSystemInputs();
        ++f;
        if(is->isButtonPressed(0) && f%2!=0) { return; }
        if(is->isButtonPressed(1) && f%4!=0) { return; }
        if(is->isButtonPressed(2) && f%8!=0) { return; }
        if(is->isButtonPressed(3) && f%16!=0){ return; }
    }
    atomicKickDraw();
    atomicWaitUntilDrawCallbackComplete();
}

void AtomicGame::frameEnd()
{
    if(atomicGetConfig()->pause) { return; }

    m_world->frameEnd();
}


void AtomicGame::drawCallback()
{
    AtomicRenderer::getInstance()->beforeDraw();
    if(m_input_server->getClassID()==IInputServer::IS_REPLAY) {
        const uvec2 &wsize = atomicGetWindowSize();
        uint32 len  = static_cast<InputServerReplay*>(m_input_server)->getReplayLength();
        uint32 pos  = static_cast<InputServerReplay*>(m_input_server)->getReplayPosition();
        char buf[128];
        istsprintf(buf, "Replay %d / %d", pos, len);
        atomicGetSystemTextRenderer()->addText(vec2(5.0f, (float32)wsize.y), buf);
    }
    if(m_world) {
        m_world->draw();
    }
    atomicDbgDebugMenuDraw();
    AtomicRenderer::getInstance()->draw();
}

SFMT* AtomicGame::getRandom()
{
    atomicDbgAssertSyncLock();
    return &m_rand;
}

void AtomicGame::handleLevelEditorCommands( const LevelEditorCommand &c )
{
    static IEntity *s_last_entity;
    if(c.type==LEC_Create) {
        const LevelEditorCommand_Create &cmd = reinterpret_cast<const LevelEditorCommand_Create&>(c);
        s_last_entity = atomicGetEntitySet()->createEntity<Enemy_Test>();
    }
    else if(c.type==LEC_Delete) {
        const LevelEditorCommand_Delete &cmd = reinterpret_cast<const LevelEditorCommand_Delete&>(c);
        IEntity *e = cmd.entity_id==uint32(-1) ? s_last_entity : atomicGetEntity(cmd.entity_id);
        if(e) {
            atomicCall(e, kill, 0);
        }
    }
    else if(c.type==LEC_Call) {
        const LevelEditorCommand_Call &cmd = reinterpret_cast<const LevelEditorCommand_Call&>(c);
        IEntity *e = cmd.entity_id==uint32(-1) ? s_last_entity : atomicGetEntity(cmd.entity_id);
        if(e) {
            e->call((FunctionID)cmd.function_id, cmd.arg);
        }
    }
}

void AtomicGame::handleLevelEditorQueries( LevelEditorQuery &cmd )
{
    if(cmd.type==LEQ_Entities) {
        jsonizeEntities(cmd.response);
    }
}

void AtomicGame::jsonizeEntities( std::string &out )
{
    m_ctx_jsonize_entities.entities.clear();
    m_world->jsonizeEntities(m_ctx_jsonize_entities);

    char buf[64];
    auto &entities = m_ctx_jsonize_entities.entities;
    out += "{\"ids\":[";
    for(size_t i=0; i<entities.size(); ++i) {
        istsprintf(buf, "%d,", entities[i].id);
        out+=buf;
    }
    out.pop_back();
    out += "], \"types\":[";
    for(size_t i=0; i<entities.size(); ++i) {
        istsprintf(buf, "%d,", entities[i].type);
        out+=buf;
    }
    out.pop_back();
    out += "], \"sizes\":[";
    for(size_t i=0; i<entities.size(); ++i) {
        istsprintf(buf, "%.2f,%.2f,", entities[i].size.x, entities[i].size.y);
        out+=buf;
    }
    out.pop_back();
    out += "], \"positions\":[";
    for(size_t i=0; i<entities.size(); ++i) {
        istsprintf(buf, "%.2f,%.2f,", entities[i].pos.x, entities[i].pos.y);
        out+=buf;
    }
    out.pop_back();
    out += "]}";
}

} // namespace atomic