#ifndef atm_Network_LevelEditorServer_h
#define atm_Network_LevelEditorServer_h
#include "externals.h"
#include "LevelEditorCommand.h"

namespace atm {

#ifdef atm_enable_LevelEditorServer

struct LevelEditorServerConfig
{
    uint16 max_queue;
    uint16 max_threads;

    LevelEditorServerConfig();
};

class LevelEditorCommandHandler;
class LevelEditorRequestHandlerFactory;

class LevelEditorServer
{
friend class LevelEditorCommandHandler;
friend class LevelEditorQueryHandler;
friend class LevelEditorRequestHandlerFactory;
public:
    enum ErrorCode {
        ER_Ok,
        ER_StartFailed,
    };

    static void initializeInstance();
    static void finalizeInstance();
    static LevelEditorServer* getInstance();

    void start();
    void stop();
    void restart();

    typedef std::function<void (const LevelEditorCommand&)> CommandHandler;
    typedef std::function<void (LevelEditorQuery&)> QueryHandler;
    void handleCommands(const CommandHandler &h);
    void handleQueries(const QueryHandler &h);


private:
    LevelEditorServer();
    ~LevelEditorServer();
    void pushCommand(const LevelEditorCommand &cmd);
    void pushQuery(LevelEditorQuery &q);
    void clearQuery();

    vec2 randomVec2();
    bool endFlag() const { return m_end_flag; }

private:
    typedef ist::vector<LevelEditorCommand> CommandCont;
    typedef ist::vector<LevelEditorQuery*> QueryCont;

    static LevelEditorServer *s_inst;
    Poco::Net::HTTPServer *m_server;
    LevelEditorServerConfig m_conf;
    SFMT m_rand;
    bool m_end_flag;

    ist::Mutex m_mutex_commands;
    CommandCont m_commands;
    CommandCont m_commands_tmp;

    ist::Mutex m_mutex_queries;
    QueryCont m_queries;
};

#define atmLevelEditorServerInitialize()     LevelEditorServer::initializeInstance()
#define atmLevelEditorServerFinalize()       LevelEditorServer::finalizeInstance()
#define atmLevelEditorHandleCommands(...)    LevelEditorServer::getInstance()->handleCommands(__VA_ARGS__)
#define atmLevelEditorHandleQueries(...)     LevelEditorServer::getInstance()->handleQueries(__VA_ARGS__)

#else // atm_enable_LevelEditorServer

#define atmLevelEditorServerInitialize()     
#define atmLevelEditorServerFinalize()       
#define atmLevelEditorHandleCommands(H)      
#define atmLevelEditorHandleQueries(H)       

#endif // atm_enable_LevelEditorServer

} // namespace atm
#endif // atm_Network_LevelEditorServer_h
