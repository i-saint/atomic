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

#include "Collision.h"
#include "Entity/Routine.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_world(NULL)
, m_input_server(NULL)
{
#ifdef atomic_enable_sync_lock
    m_sync_lock = false;
#endif // atomic_enable_sync_lock
    MessageRouter::initializeInstance();

    wchar_t name[16] = L"test";
    m_input_server = CreateInputServerLocal();
    m_input_server->addPlayer(0, name, 0);

    m_world = istNew(World)();
    m_world->initialize();

    // 今回は固定値で初期化
    m_rand.initialize(0);
}

AtomicGame::~AtomicGame()
{
    if(m_input_server->getTypeID()==IInputServer::IS_Local && atomicGetConfig()->output_replay)
    {
        char path[128];
        char date[128];
        CreateDateString(date, _countof(date));
        istSPrintf(path, "%s.replay", date);
        for(size_t i=0; i<_countof(path); ++i) { if(path[i]=='/' || path[i]==':') { path[i]='-'; } }
        m_input_server->save(path);
    }

    istSafeDelete(m_world);
    istSafeDelete(m_input_server);

    MessageRouter::finalizeInstance();
}

bool AtomicGame::readReplayFromFile(const char *path)
{
    IInputServer *ris = CreateInputServerReplay();
    if(ris->load(path)) {
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
    if(!atomicDbgDebugMenuIsActive()) {
        m_input_server->pushInput(0, *atomicGetSystemInputs());
    }
    else {
        m_input_server->pushInput(0, InputState());
    }

    LevelEditorServer::getInstance()->handleCommands(std::bind(&IInputServer::pushLevelEditorCommand, m_input_server, std::placeholders::_1));
    LevelEditorServer::getInstance()->handleQueries(std::bind(&AtomicGame::handleLevelEditorQueries, this, std::placeholders::_1) );
    m_input_server->update();
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

    if(m_input_server->getTypeID()==IInputServer::IS_Replay) {
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
    if(m_input_server->getTypeID()==IInputServer::IS_Replay) {
        const uvec2 &wsize = atomicGetWindowSize();
        uint32 len  = m_input_server->getPlayLength();
        uint32 pos  = m_input_server->getPlayPosition();
        char buf[128];
        istSPrintf(buf, "Replay %d / %d", pos, len);
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
        IEntity *e = atomicGetEntitySet()->createEntity<Enemy_Test>();
        s_last_entity = e;
        atomicCall(e, setCollisionShape, CS_SPHERE);
        atomicCall(e, setModel, PSET_SPHERE_SMALL);
        atomicCall(e, setPosition, GenRandomVector2() * 2.2f);
        atomicCall(e, setLife, 15.0f);
        atomicCall(e, setAxis1, GenRandomUnitVector3());
        atomicCall(e, setAxis2, GenRandomUnitVector3());
        atomicCall(e, setRotateSpeed1, 2.4f);
        atomicCall(e, setRotateSpeed2, 2.4f);
        atomicCall(e, setRoutine, ROUTINE_HOMING_PLAYER);
        atomicCall(e, setLightRadius, 0.5f);
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
        IEntity *e = cmd.entity_id==0 ? s_last_entity : atomicGetEntity(cmd.entity_id);
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
        istSPrintf(buf, "%d,", entities[i].id);
        out+=buf;
    }
    out.pop_back();
    out += "], \"types\":[";
    for(size_t i=0; i<entities.size(); ++i) {
        istSPrintf(buf, "%d,", entities[i].type);
        out+=buf;
    }
    out.pop_back();
    out += "], \"sizes\":[";
    for(size_t i=0; i<entities.size(); ++i) {
        istSPrintf(buf, "%.2f,%.2f,", entities[i].size.x, entities[i].size.y);
        out+=buf;
    }
    out.pop_back();
    out += "], \"positions\":[";
    for(size_t i=0; i<entities.size(); ++i) {
        istSPrintf(buf, "%.2f,%.2f,", entities[i].pos.x, entities[i].pos.y);
        out+=buf;
    }
    out.pop_back();
    out += "]}";
}

} // namespace atomic