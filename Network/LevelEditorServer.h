#ifndef atomic_Network_LevelEditorServer_h
#define atomic_Network_LevelEditorServer_h
#include "externals.h"
#include "LevelEditorCommand.h"

namespace atomic {


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

    typedef std::function<void (const LevelEditorCommand&)> CommandProcessor;
    typedef std::function<void (LevelEditorQuery&)> QueryProcessor;
    void handleCommands(const CommandProcessor &proc);
    void handleQueries(const QueryProcessor &proc);


private:
    LevelEditorServer();
    ~LevelEditorServer();
    void pushCommand(const LevelEditorCommand &cmd);
    void pushQuery(LevelEditorQuery &q);
    void clearQuery();

    vec2 randomVec2();

private:
    typedef ist::vector<LevelEditorCommand> CommandCont;
    typedef ist::vector<LevelEditorQuery*> QueryCont;

    static LevelEditorServer *s_inst;
    Poco::Net::HTTPServer *m_server;
    LevelEditorServerConfig m_conf;
    SFMT m_rand;
    bool m_accept_request;

    ist::Mutex m_mutex_commands;
    CommandCont m_commands;
    CommandCont m_commands_tmp;

    ist::Mutex m_mutex_queries;
    QueryCont m_queries;
};

} // namespace atomic
#endif // atomic_Network_LevelEditorServer_h
