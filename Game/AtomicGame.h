#ifndef atm_Game_AtomicGame_h
#define atm_Game_AtomicGame_h
#include "Input.h"
#include "AtomicApplication.h"
#include "Network/LevelEditorCommand.h"
#include "Network/InputServer.h"
#include "Network/Protocol.h"


namespace atm {

class World;
class AtomicRenderer;

struct GameStartConfig
{
    enum GameMode {
        GM_Campaign,
        GM_Horde,
        GM_Replay,
        GM_Edit,
    };
    enum NetworkMode {
        NM_Server,
        NM_Client,
        NM_Offline,
    };
    GameMode    gmode;
    NetworkMode nmode;
    std::string path_to_level;
    std::string path_to_replay;
    std::string server_address;
    uint16      server_port;

    GameStartConfig()
        : gmode(GM_Campaign)
        , nmode(NM_Offline)
        , server_address("localhost")
        , server_port(10050)
    {
    }
};


class atmAPI AtomicGame
{
public:
    AtomicGame();
    ~AtomicGame();

    bool config(const GameStartConfig &conf);
    bool readReplayFromFile(const char *path);

    void frameBegin();
    void update(float32 dt);
    void asyncupdateBegin(float32 dt);
    void asyncupdateEnd();
    void draw();
    void frameEnd();

    // 描画スレッドから呼ばれる
    void drawCallback();

    void handleLevelEditorCommands(const LevelEditorCommand &cmd);
    void handleLevelEditorQueries(LevelEditorQuery &cmd);
    void handlePMessages(const PMessage &mes);
    int handleCommandLine(const stl::wstring &command);
    void handleStateQuery(std::string &out);
    void handleEntityQuery(std::string &out, EntityHandle h);

    const InputState& getIngameInputs() const { return m_input_server->getInput(0); }
    World* getWorld()   { return m_world; }
    SFMT* getRandom();
    uint32 getFrame() const                 { return m_frame; }
    PlayerID getPlayerID() const            { return m_player_id; }
    bool isUpdateSkipped() const            { return m_skip_update; }
    bool isDrawSkipped() const              { return m_skip_draw; }
    bool isWaitVSyncRequired() const        { return !isUpdateSkipped() && !isDrawSkipped(); }
    bool isEditMode() const                 { return m_is_edit; }

    bool serialize(std::ostream &st);
    bool deserialize(std::istream &st);

    void testSerialize();
    void testDeserialize();

#ifdef atm_enable_sync_lock
    void dbgLockSyncMethods()               { m_sync_lock=true; }
    void dbgUnlockSyncMethods()             { m_sync_lock=false; }
    bool dbgIsSyncMethodsEnabled() const    { return m_sync_lock; }
#endif // atm_enable_sync_lock

private:
    IInputServer    *m_input_server;
    World           *m_world;
    SFMT            m_rand;
    uint32          m_frame;
    float32         m_resource; // RTS モードでユニットを生成するためのコスト。毎秒 1 追加 & 敵キャラ倒すと追加
    PlayerID        m_player_id;
    EntitiesQueryContext m_ctx_entities_query;
    bool m_skip_update;
    bool m_skip_draw;
    bool m_is_edit;
#ifdef atm_enable_sync_lock
    bool m_sync_lock;
#endif // atm_enable_sync_lock
};

#define atmGetWorld()           atmGetGame()->getWorld()
#define atmGetIngameInputs()    atmGetGame()->getIngameInputs()
#define atmGetRandom()          atmGetGame()->getRandom()
#define atmGetFrame()           atmGetGame()->getFrame()

bool atmIsEditMode();

#ifdef atm_enable_sync_lock
#   define  atmDbgLockSyncMethods()      atmGetGame()->dbgLockSyncMethods()
#   define  atmDbgUnlockSyncMethods()    atmGetGame()->dbgUnlockSyncMethods()
#   define  atmDbgAssertSyncLock()       istAssert(!atmGetGame()||!atmGetGame()->dbgIsSyncMethodsEnabled())
#else
#   define  atmDbgLockSyncMethods()      
#   define  atmDbgUnlockSyncMethods()    
#   define  atmDbgAssertSyncLock()       
#endif // atm_enable_sync_lock

} // namespace atm
#endif atm_Game_AtomicGame_h
