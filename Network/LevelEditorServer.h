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
};

struct LevelEditorCommand
{
    LevelEditorCommandID command;
};

struct LevelEditorCommand_Create
{
    LevelEditorCommandID command;
    uint32 entity_typeid;
};

struct LevelEditorCommand_Delete
{
    LevelEditorCommandID command;
    uint32 entity_typeid;
};

struct LevelEditorCommand_Call
{
    LevelEditorCommandID command;
    uint32 function_id;
    variant arg;
};



struct LevelEditorServerConfig
{
    uint16 port;
    uint16 max_queue;
    uint16 max_threads;

    LevelEditorServerConfig();
};

class LevelEditorServer
{
public:
    static void initializeInstance();
    static void finalizeInstance();
    static LevelEditorServer* getInstance();

    void start();
    void stop();
    void restart();

    typedef std::function<void (const LevelEditorCommand*)> CommandProcessor;
    void handleCommands(const CommandProcessor &proc);


private:
    typedef stl::vector<variant32> CommandCont;

    LevelEditorServer();
    ~LevelEditorServer();

    static LevelEditorServer *s_inst;
    Poco::Net::HTTPServer *m_server;
    LevelEditorServerConfig m_conf;
    CommandCont m_commands;
    CommandCont m_commands_tmp;
};

} // namespace atomic
#endif // atomic_Network_LevelEditorServer_h
