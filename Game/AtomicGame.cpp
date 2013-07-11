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
#include "Game/EntityModule.h"
#include "Game/EntityQuery.h"
#include "Network/GameCLient.h"
#include "Network/LevelEditorServer.h"

#include "CollisionModule.h"
#include "Entity/Routine.h"
#include <zip_stream/zipstream.hpp>

namespace atm {


AtomicGame::AtomicGame()
: m_input_server(NULL)
, m_world(NULL)
, m_frame(0)
, m_resource(0.0f)
, m_skip_update(false)
{
    wdmAddNode("Game/testSerialize()", &AtomicGame::testSerialize, this);
    wdmAddNode("Game/testDeserialize()", &AtomicGame::testDeserialize, this);
}

AtomicGame::~AtomicGame()
{
    if(atmGetConfig()->output_replay)
    {
        char path[128];
        char date[128];
        CreateDateString(date, _countof(date));
        for(size_t i=0; i<_countof(date); ++i) { if(date[i]=='/' || date[i]==':') { date[i]='-'; } }
        istSPrintf(path, "Replay/%s.replay", date);
        m_input_server->save(path);
    }

    istSafeDelete(m_world);
    istSafeDelete(m_input_server);

    MessageRouter::finalizeInstance();
    wdmEraseNode("Game");
}

bool AtomicGame::config(const GameStartConfig &conf)
{
#ifdef atm_enable_sync_lock
    m_sync_lock = false;
#endif // atm_enable_sync_lock
    MessageRouter::initializeInstance();

    PlayerName name = L"test";
    if(conf.gmode==GameStartConfig::GM_Replay) {
        readReplayFromFile(conf.path_to_replay.c_str());
    }
    else if(conf.nmode==GameStartConfig::NM_Offline) {
        m_input_server = CreateInputServerLocal();
        m_input_server->addPlayer(0, name, 0);
    }
    else if(conf.nmode==GameStartConfig::NM_Server || conf.nmode==GameStartConfig::NM_Client) {
        m_input_server = CreateInputServerNetwork();
        m_input_server->addPlayer(0, name, 0);
    }

    m_world = istNew(World)();
    m_world->initialize();

    // 今回は固定値で初期化
    m_rand.initialize(0);
    return true;
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
        m_input_server->pushInput(0, atmGetSystemInputs()->getRawInput());
    }
    m_resource += dt;

    atmLevelEditorHandleCommands( std::bind(&IInputServer::pushLevelEditorCommand, m_input_server, std::placeholders::_1));
    atmLevelEditorHandleQueries( std::bind(&AtomicGame::handleLevelEditorQueries, this, std::placeholders::_1) );
    atmGameClientHandleMessages( std::bind(&AtomicGame::handlePMessages, this, std::placeholders::_1) );

    m_skip_update = atmGetConfig()->pause || !m_input_server->sync();
    if(!m_skip_update)
    {
        m_input_server->update();
        m_world->update(1.0f);
    }
}

void AtomicGame::asyncupdateBegin(float32 dt)
{
    if(m_skip_update) { return; }

    atmDbgLockSyncMethods();
    m_world->asyncupdateBegin(dt);
}

void AtomicGame::asyncupdateEnd()
{
    if(m_skip_update) { return; }

    m_world->asyncupdateEnd();
    atmDbgUnlockSyncMethods();
}


void AtomicGame::draw()
{
    if(m_skip_update) { return; }
    // todo: フレームスキップ処理

    m_skip_draw = false;
    if(m_input_server->getTypeID()==IInputServer::IS_Replay) {
        static uint32 f;
        const InputState *is = atmGetSystemInputs();
        ++f;
        if(is->isButtonPressed(0) && f%2!=0) { m_skip_draw=true; return; }
        if(is->isButtonPressed(1) && f%4!=0) { m_skip_draw=true; return; }
        if(is->isButtonPressed(2) && f%8!=0) { m_skip_draw=true; return; }
        if(is->isButtonPressed(3) && f%16!=0){ m_skip_draw=true; return; }
    }
}

void AtomicGame::frameEnd()
{
    if(m_skip_update) { return; }

    m_world->frameEnd();
    ++m_frame;
}


void AtomicGame::drawCallback()
{
    if(m_input_server->getTypeID()==IInputServer::IS_Replay) {
        const uvec2 &wsize = atmGetWindowSize();
        uint32 len  = m_input_server->getPlayLength();
        uint32 pos  = m_input_server->getPlayPosition();
        char buf[128];
        istSPrintf(buf, "Replay %d / %d", pos, len);
        atmGetTextRenderer()->addText(vec2(5.0f, (float32)wsize.y), buf);
    }

#ifdef atm_enable_GameClient
    if(auto *client=atmGameClientGet()) {
        uint32 i = 0;
        auto &clients = client->getClientStates();
        for(auto it=clients.begin(); it!=clients.end(); ++it) {
            auto &stat = it->second;
            wchar_t buf[128];
            istSPrintf(buf, L"%s - ping %d", stat.name, stat.ping);
            atmGetTextRenderer()->addText(vec2(5.0f, 20.0f*i + 80.0f), buf);
            ++i;
        }
    }
#endif // atm_enable_GameClient

    if(m_world) {
        m_world->draw();
    }
}

SFMT* AtomicGame::getRandom()
{
    atmDbgAssertSyncLock();
    return &m_rand;
}

void AtomicGame::handleLevelEditorCommands( const LevelEditorCommand &c )
{
    static IEntity *s_last_entity;
    if(c.type==LEC_Create) {
        const LevelEditorCommand_Create &cmd = reinterpret_cast<const LevelEditorCommand_Create&>(c);
        EntityClassID ecid = (EntityClassID)cmd.classid;
        EntityClassInfo *eci = GetEntityClassInfo(ecid);
        bool ok = false;
        if(atmIsEditMode()) { ok=true; }
        else {
            if(m_resource>=eci->cost) {
                m_resource-=eci->cost;
                ok = true;
            }
        }

        if(ok) {
            IEntity *e = atmCreateEntity(ecid);
            s_last_entity = e;
        }
    }
    else if(c.type==LEC_Delete) {
        const LevelEditorCommand_Delete &cmd = reinterpret_cast<const LevelEditorCommand_Delete&>(c);
        IEntity *e = cmd.entity_id==0 ? s_last_entity : atmGetEntity(cmd.entity_id);
        if(e) {
            atmCall(e, kill, 0);
        }
    }
    else if(c.type==LEC_Call) {
        const LevelEditorCommand_Call &cmd = reinterpret_cast<const LevelEditorCommand_Call&>(c);
        IEntity *e = cmd.entity==0 ? s_last_entity : atmGetEntity(cmd.entity);
        if(e) {
            e->call((FunctionID)cmd.function, &cmd.arg);
        }
    }
}

void AtomicGame::handleLevelEditorQueries( LevelEditorQuery &cmd )
{
    if(cmd.type==LEQ_Entities) {
        handleEntitiesQuery(cmd.response);
    }
    else if(cmd.type==LEQ_Entity) {
        handleEntityQuery(cmd.response, (EntityHandle)cmd.optional);
    }
}

void AtomicGame::handleEntitiesQuery( std::string &out )
{
    EntitiesQueryContext &eqc = m_ctx_entities_query;
    eqc.clear();
    m_world->handleEntitiesQuery(eqc);

#ifdef atm_enable_WebGL
    uint32 wpos = 0;

    uint32 num_entities = (uint32)eqc.id.size();
    uint32 num_bullets = (uint32)eqc.bullets.size();
    uint32 num_lasers = (uint32)eqc.lasers.size();
    uint32 num_fluids = (uint32)eqc.fluids.size();
    out.resize(sizeof(uint32)*4+eqc.sizeByte());

    *(uint32*)(&out[wpos]) = num_entities;
    wpos += sizeof(uint32);
    *(uint32*)(&out[wpos]) = num_bullets;
    wpos += sizeof(uint32);
    *(uint32*)(&out[wpos]) = num_lasers;
    wpos += sizeof(uint32);
    *(uint32*)(&out[wpos]) = num_fluids;
    wpos += sizeof(uint32);

    if(num_entities) {

        memcpy(&out[wpos], &eqc.id[0], sizeof(eqc.id)*num_entities);
        wpos += sizeof(uint32)*num_entities;
        memcpy(&out[wpos], &eqc.trans[0], sizeof(mat4)*num_entities);
        wpos += sizeof(mat4)*num_entities;
        memcpy(&out[wpos], &eqc.size[0], sizeof(vec3)*num_entities);
        wpos += sizeof(vec3)*num_entities;
        memcpy(&out[wpos], &eqc.color[0], sizeof(vec4)*num_entities);
        wpos += sizeof(vec4)*num_entities;
    }
    if(num_bullets) {
        memcpy(&out[wpos], &eqc.bullets[0], sizeof(vec2)*num_bullets);
        wpos += sizeof(vec2)*num_bullets;
    }
    if(num_lasers) {
        memcpy(&out[wpos], &eqc.lasers[0], sizeof(vec3)*num_lasers);
        wpos += sizeof(vec3)*num_lasers;
    }
    if(num_fluids) {
        memcpy(&out[wpos], &eqc.fluids[0], sizeof(vec2)*num_fluids);
        wpos += sizeof(vec2)*num_lasers;
    }

#else // atm_enable_WebGL

    char buf[64];
    auto &ids = eqc.id;
    auto &types = eqc.type;
    auto &sizes = eqc.size;
    auto &positions = eqc.pos;
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
#endif // atm_enable_WebGL
}

void AtomicGame::handleEntityQuery( std::string &out, EntityHandle h )
{
    out += "[";
    if(IEntity *e=atmGetEntity(h)) {
        e->jsonize(out);
        out.pop_back();
    }
    out += "]";
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

bool AtomicGame::serialize( std::ostream &st )
{
    try {
        boost::archive::binary_oarchive ar(st);
        ar & m_world;
    }
    catch(std::exception &e) {
        istPrint(e.what());
    }
    return true;
}

bool AtomicGame::deserialize( std::istream &st )
{
    istSafeDelete(m_world);
    try {
        boost::archive::binary_iarchive ar(st);
        ar & m_world;
    }
    catch(std::exception &e) {
        istPrint(e.what());
    }
    return true;
}

void AtomicGame::testSerialize()
{
    std::ofstream fs("state.atbin", std::ios::binary);
    zlib_stream::zip_ostream zipper(fs);
    serialize(zipper);
}

void AtomicGame::testDeserialize()
{
    std::ifstream fs("state.atbin", std::ios::binary);
    zlib_stream::zip_istream zipper(fs);
    deserialize(zipper);
}

} // namespace atm
