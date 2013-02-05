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


union istAlign(16) LevelEditorCommand
{
    struct {
        LEC_Type type;
        uint32 frame;
    };
    uint32 dummy[8];

    LevelEditorCommand() : type(LEC_Unknown), frame(0)
    {
        std::fill_n(dummy, _countof(dummy), 0);
    }
};

struct istAlign(16) LevelEditorCommand_Create
{
    LEC_Type type;
    uint32 frame;
    uint32 entity_typeid;
    uint32 dummy[5];

    LevelEditorCommand_Create() : type(LEC_Create), frame(0), entity_typeid(0)
    {
        std::fill_n(dummy, _countof(dummy), 0);
    }
};
BOOST_STATIC_ASSERT(sizeof(LevelEditorCommand_Create)<=sizeof(LevelEditorCommand));

struct istAlign(16) LevelEditorCommand_Delete
{
    LEC_Type type;
    uint32 frame;
    uint32 entity_id;
    uint32 dummy[5];

    LevelEditorCommand_Delete() : type(LEC_Delete), frame(0), entity_id(0)
    {
        std::fill_n(dummy, _countof(dummy), 0);
    }
};
BOOST_STATIC_ASSERT(sizeof(LevelEditorCommand_Delete)<=sizeof(LevelEditorCommand));

struct istAlign(16) LevelEditorCommand_Call
{
    LEC_Type type;
    uint32 frame;
    uint32 entity_id;
    uint32 function_id;
    variant arg;

    LevelEditorCommand_Call() : type(LEC_Call) {}
};
BOOST_STATIC_ASSERT(sizeof(LevelEditorCommand_Call)<=sizeof(LevelEditorCommand));


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
