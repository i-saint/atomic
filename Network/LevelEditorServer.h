#ifndef atomic_Network_LevelEditorServer_h
#define atomic_Network_LevelEditorServer_h
#include "externals.h"
#include "LevelEditorCommand.h"

namespace atomic {

#ifdef atomic_enable_LevelEditorServer

struct LevelEditorServerConfig
{
    uint16 port;
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

#define atomicLevelEditorServerInitialize()     LevelEditorServer::initializeInstance()
#define atomicLevelEditorServerFinalize()       LevelEditorServer::finalizeInstance()
#define atomicLevelEditorHandleCommands(...)    LevelEditorServer::getInstance()->handleCommands(__VA_ARGS__)
#define atomicLevelEditorHandleQueries(...)     LevelEditorServer::getInstance()->handleQueries(__VA_ARGS__)

#else // atomic_enable_LevelEditorServer

#define atomicLevelEditorServerInitialize()     
#define atomicLevelEditorServerFinalize()       
#define atomicLevelEditorHandleCommands(H)      
#define atomicLevelEditorHandleQueries(H)       

#endif // atomic_enable_LevelEditorServer

} // namespace atomic
#endif // atomic_Network_LevelEditorServer_h
