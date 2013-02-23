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
#include "Network/GameCLient.h"
#include "Network/LevelEditorServer.h"

#include "Collision.h"
#include "Entity/Routine.h"

namespace atomic {


AtomicGame::AtomicGame()
: m_input_server(NULL)
, m_world(NULL)
, m_frame(0)
, m_skip_update(false)
{
#ifdef atomic_enable_sync_lock
    m_sync_lock = false;
#endif // atomic_enable_sync_lock
    MessageRouter::initializeInstance();

    PlayerName name = L"test";
    //m_input_server = CreateInputServerLocal();
    m_input_server = CreateInputServerNetwork();
    m_input_server->addPlayer(0, name, 0);

    m_world = istNew(World)();
    m_world->initialize();

    // 今回は固定値で初期化
    m_rand.initialize(0);
}

AtomicGame::~AtomicGame()
{
    if(atomicGetConfig()->output_replay)
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
    if(m_skip_update) { return; }

    m_world->frameBegin();
}

void AtomicGame::update(float32 dt)
{
    if(!m_skip_update) {
        if(!atomicDbgDebugMenuIsActive()) {
            m_input_server->pushInput(0, atomicGetSystemInputs()->getRawInput());
        }
        else {
            m_input_server->pushInput(0, RepInput());
        }
    }

    istCommandlineFlush();
    atomicLevelEditorHandleCommands( std::bind(&IInputServer::pushLevelEditorCommand, m_input_server, std::placeholders::_1));
    atomicLevelEditorHandleQueries( std::bind(&AtomicGame::handleLevelEditorQueries, this, std::placeholders::_1) );
    atomicGameClientHandleMessages( std::bind(&AtomicGame::handlePMessages, this, std::placeholders::_1) );

    m_skip_update = atomicGetConfig()->pause || !m_input_server->sync();
    if(!m_skip_update)
    {
        m_input_server->update();
        m_world->update(1.0f);
    }
}

void AtomicGame::asyncupdateBegin(float32 dt)
{
    if(m_skip_update) { return; }

    atomicDbgLockSyncMethods();
    m_world->asyncupdateBegin(dt);
}

void AtomicGame::asyncupdateEnd()
{
    if(m_skip_update) { return; }

    m_world->asyncupdateEnd();
    atomicDbgUnlockSyncMethods();
}


void AtomicGame::draw()
{
    if(m_skip_update) { return; }
    // todo: フレームスキップ処理

    m_skip_draw = false;
    if(m_input_server->getTypeID()==IInputServer::IS_Replay) {
        static uint32 f;
        const InputState *is = atomicGetSystemInputs();
        ++f;
        if(is->isButtonPressed(0) && f%2!=0) { m_skip_draw=true; return; }
        if(is->isButtonPressed(1) && f%4!=0) { m_skip_draw=true; return; }
        if(is->isButtonPressed(2) && f%8!=0) { m_skip_draw=true; return; }
        if(is->isButtonPressed(3) && f%16!=0){ m_skip_draw=true; return; }
    }
    atomicKickDraw();
    atomicWaitUntilDrawCallbackComplete();
}

void AtomicGame::frameEnd()
{
    if(m_skip_update) { return; }

    m_world->frameEnd();
    ++m_frame;
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

    if(auto *client=atomicGameClientGet()) {
        uint32 i = 0;
        auto &clients = client->getClientStates();
        for(auto it=clients.begin(); it!=clients.end(); ++it) {
            auto &stat = it->second;
            wchar_t buf[128];
            istSPrintf(buf, L"%s - ping %d", stat.name, stat.ping);
            atomicGetSystemTextRenderer()->addText(vec2(5.0f, 20.0f*i + 80.0f), buf);
            ++i;
        }
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
        IEntity *e = atomicCreateEntity(Enemy_Test);
        s_last_entity = e;
        atomicCall(e, setCollisionShape, CS_Sphere);
        atomicCall(e, setModel, PSET_SPHERE_SMALL);
        atomicCall(e, setPosition, GenRandomVector2() * 2.2f);
        atomicCall(e, setLife, 15.0f);
        atomicCall(e, setAxis1, GenRandomUnitVector3());
        atomicCall(e, setAxis2, GenRandomUnitVector3());
        atomicCall(e, setRotateSpeed1, 2.4f);
        atomicCall(e, setRotateSpeed2, 2.4f);
        atomicCall(e, setRoutine, RCID_Routine_HomingPlayer);
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
        handleEntitiesQuery(cmd.response);
    }
}

void AtomicGame::handleEntitiesQuery( std::string &out )
{
    m_ctx_entities_query.clear();
    m_world->handleEntitiesQuery(m_ctx_entities_query);

#ifdef atomic_enable_BinaryEntityData

    uint32 num_entities = m_ctx_entities_query.id.size();
    out.resize(sizeof(uint32)+m_ctx_entities_query.sizeByte());
    *(uint32*)(&out[0]) = m_ctx_entities_query.id.size();

    uint32 wpos = sizeof(uint32);
    memcpy(&out[wpos], &m_ctx_entities_query.id[0], sizeof(uint32)*num_entities);
    wpos += sizeof(uint32)*num_entities;
    memcpy(&out[wpos], &m_ctx_entities_query.type[0], sizeof(uint32)*num_entities);
    wpos += sizeof(uint32)*num_entities;
    memcpy(&out[wpos], &m_ctx_entities_query.size[0], sizeof(vec2)*num_entities);
    wpos += sizeof(vec2)*num_entities;
    memcpy(&out[wpos], &m_ctx_entities_query.pos[0], sizeof(vec2)*num_entities);
    wpos += sizeof(vec2)*num_entities;

#else // atomic_enable_BinaryEntityData

    char buf[64];
    auto &ids = m_ctx_entities_query.id;
    auto &types = m_ctx_entities_query.type;
    auto &sizes = m_ctx_entities_query.size;
    auto &positions = m_ctx_entities_query.pos;
    out += "{\"ids\":[";
    for(size_t i=0; i<ids.size(); ++i) {
        istSPrintf(buf, "%d,", ids[i]);
        out+=buf;
    }
    out.pop_back();
    out += "], \"types\":[";
    for(size_t i=0; i<types.size(); ++i) {
        istSPrintf(buf, "%d,", types[i]);
        out+=buf;
    }
    out.pop_back();
    out += "], \"sizes\":[";
    for(size_t i=0; i<sizes.size(); ++i) {
        istSPrintf(buf, "%.2f,%.2f,", sizes[i].x, sizes[i].y);
        out+=buf;
    }
    out.pop_back();
    out += "], \"positions\":[";
    for(size_t i=0; i<positions.size(); ++i) {
        istSPrintf(buf, "%.2f,%.2f,", positions[i].x, positions[i].y);
        out+=buf;
    }
    out.pop_back();
    out += "]}";
#endif // atomic_enable_BinaryEntityData
}

void AtomicGame::handlePMessages( const PMessage &mes )
{
    m_input_server->handlePMessage(mes);

    switch(mes.type) {
    case PM_Accepted:
        {
            auto &m = reinterpret_cast<const PMessage_Accepted&>(mes);
            m_player_id = m.player_id;
        }
        break;
    case PM_Rejected:
        {
            istAssert(false);
        }
        break;
    }
}

} // namespace atomic