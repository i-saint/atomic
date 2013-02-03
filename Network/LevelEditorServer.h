#ifndef atomic_Network_LevelEditorServer_h
#define atomic_Network_LevelEditorServer_h
#include "externals.h"

namespace atomic {

// Level Editor Commans
enum LEC_Type
{
    LEC_Unknown,
    LEC_Create,
    LEC_Delete,
    LEC_Call,
};

// Level Editor Query
enum LEQ_Type
{
    LEQ_Unknown,
    LEQ_Entities,
    LEQ_Entity,
    LEQ_Players,
};


struct LevelEditorCommand
{
    LEC_Type type;

    LevelEditorCommand() : type(LEC_Unknown) {}
};

struct LevelEditorCommand_Create
{
    LEC_Type type;
    uint32 entity_typeid;

    LevelEditorCommand_Create() : type(LEC_Create) {}
};

struct LevelEditorCommand_Delete
{
    LEC_Type type;
    uint32 entity_id;

    LevelEditorCommand_Delete() : type(LEC_Delete) {}
};

struct LevelEditorCommand_Call
{
    LEC_Type type;
    uint32 entity_id;
    uint32 function_id;
    uint32 dummy;
    variant arg;

    LevelEditorCommand_Call() : type(LEC_Call) {}
};

struct LevelEditorQuery
{
    LEQ_Type type;
    uint32 optional;

    std::string response;
    bool completed;

    LevelEditorQuery()
        : optional(0)
        , completed(false)
    {}
};



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
    void pushCommand(const variant32 &cmd);
    void pushQuery(LevelEditorQuery &q);
    void clearQuery();

private:
    typedef stdex::vector<variant32> CommandCont;
    typedef stdex::vector<LevelEditorQuery*> QueryCont;

    static LevelEditorServer *s_inst;
    Poco::Net::HTTPServer *m_server;
    LevelEditorServerConfig m_conf;

    ist::Mutex m_mutex_commands;
    CommandCont m_commands;
    CommandCont m_commands_tmp;

    ist::Mutex m_mutex_queries;
    QueryCont m_queries;
    QueryCont m_queries_tmp;
};

} // namespace atomic
#endif // atomic_Network_LevelEditorServer_h
