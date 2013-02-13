#include "istPCH.h"
#include "Commandline_ConsoleFrontend.h"

namespace ist {

CommandlineConsole * CommandlineConsole::s_inst;

void CommandlineConsole::initializeInstance()
{
    if(!s_inst) {
        s_inst = istNew(CommandlineConsole)();
    }
}

void CommandlineConsole::finalizeInstance()
{
    if(s_inst) {
        istSafeDelete(s_inst);
    }
}

CommandlineConsole* CommandlineConsole::getInstance()
{
    return s_inst;
}

CommandlineConsole::CommandlineConsole()
    : m_thread(NULL)
    , m_stop(false)
#if defined(ist_env_Windows)
    , m_cs_in(NULL)
    , m_cs_out(NULL)
#endif
{
    m_thread = istNew(FunctorThread<>)(std::bind(&CommandlineConsole::dialogLoop, this));
}

CommandlineConsole::~CommandlineConsole()
{
    m_stop = true;
#if defined(ist_env_Windows)
    if(m_cs_in) {
        ::CloseHandle(m_cs_in);
    }
#endif
    m_thread->join();
    istSafeDelete(m_thread);
}

void CommandlineConsole::dialogLoop()
{
#if defined(ist_env_Windows)

    ::AllocConsole();
    m_cs_in = ::GetStdHandle(STD_INPUT_HANDLE);
    m_cs_out = ::GetStdHandle(STD_OUTPUT_HANDLE);

    DWORD num_read;
    char buf[512];
    while(!m_stop) {
        if(::ReadConsoleA(m_cs_in, &buf, _countof(buf)-1, &num_read, NULL) && num_read>0) {
            buf[num_read-1] = '\0'; // buf[num_read-1] は '\r'
            Commandline::getInstance()->pushCommand(buf);
        }
    }
    ::FreeConsole();

#elif defined(ist_env_Android)

#endif // 
}

} // namespace ist
