#ifndef atomic_Network_LevelEditorServer_h
#define atomic_Network_LevelEditorServer_h
#include "externals.h"

namespace atomic {

enum LevelEditorCommandID
{
    LEC_Unknown,
    LEC_Create,
    LEC_Delete,
    LEC_Call,
    LEC_Query,
};

struct LevelEditorCommand
{
    LevelEditorCommandID command;

    LevelEditorCommand() : command(LEC_Unknown) {}
};

struct LevelEditorCommand_Create
{
    LevelEditorCommandID command;
    uint32 entity_typeid;

    LevelEditorCommand_Create() : command(LEC_Create) {}
};

struct LevelEditorCommand_Delete
{
    LevelEditorCommandID command;
    uint32 entity_id;

    LevelEditorCommand_Delete() : command(LEC_Delete) {}
};

struct LevelEditorCommand_Call
{
    LevelEditorCommandID command;
    uint32 entity_id;
    uint32 function_id;
    uint32 dummy;
    variant arg;

    LevelEditorCommand_Call() : command(LEC_Call) {}
};

struct LevelEditorCommand_Query
{
    enum QueryID {
        Q_Entities,
        Q_Entity,
        Q_Players,
    };

    LevelEditorCommandID command;
    QueryID query;
    uint32 optional;

    LevelEditorCommand_Query() : command(LEC_Query) {}
};



struct LevelEditorServerConfig
{
    uint16 port;
    uint16 max_queue;
    uint16 max_threads;

    LevelEditorServerConfig();
};

class LevelEditorRequestHandler;
class LevelEditorRequestHandlerFactory;

class LevelEditorServer
{
friend class LevelEditorRequestHandler;
friend class LevelEditorRequestHandlerFactory;
public:
    static void initializeInstance();
    static void finalizeInstance();
    static LevelEditorServer* getInstance();

    void start();
    void stop();
    void restart();

    typedef std::function<void (const LevelEditorCommand&)> CommandProcessor;
    void handleCommands(const CommandProcessor &proc);


private:
    LevelEditorServer();
    ~LevelEditorServer();
    void pushCommand(const variant32 &cmd);

private:
    typedef stdex::vector<variant32> CommandCont;

    static LevelEditorServer *s_inst;
    Poco::Net::HTTPServer *m_server;
    LevelEditorServerConfig m_conf;
    ist::Mutex m_mutex;
    CommandCont m_commands;
    CommandCont m_commands_tmp;
};

} // namespace atomic
#endif // atomic_Network_LevelEditorServer_h
