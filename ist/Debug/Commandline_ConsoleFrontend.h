#ifndef ist_Debug_Commandline_ConsoleFrontend_h
#define ist_Debug_Commandline_ConsoleFrontend_h
#include "Commandline.h"
#include "ist/Concurrency/Thread.h"

namespace ist {

class CommandlineConsole
{
public:
    static void initializeInstance();
    static void finalizeInstance();
    static CommandlineConsole* getInstance();

private:
    CommandlineConsole();
    ~CommandlineConsole();
    void dialogLoop();

    static CommandlineConsole *s_inst;
    Thread *m_thread;
    bool m_stop;
#if defined(ist_env_Windows)
    HANDLE m_cs_out;
    HANDLE m_cs_in;
#else
#endif
};

#ifdef ist_enable_Commandline
#define istCommandlineConsoleInitialize()   ist::CommandlineConsole::initializeInstance()
#define istCommandlineConsoleFinalize()     ist::CommandlineConsole::finalizeInstance()
#else // ist_enable_Commandline
#define istCommandlineConsoleInitialize()   
#define istCommandlineConsoleFinalize()     
#endif // ist_enable_Commandline

} // namespace ist
#endif // ist_Debug_Commandline_ConsoleFrontend_h
