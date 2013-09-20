#ifndef ist_Debug_Commandline_h
#define ist_Debug_Commandline_h
#include "ist/Base/Types.h"
#include "ist/Concurrency/Mutex.h"
#include "CommandlineCommand.h"

namespace ist {


class istAPI Commandline
{
istNonCopyable(Commandline);
public:
    typedef stl::map<stl::string, ICLCommand*> CommandCont;
    typedef stl::vector<stl::string> CommandQueue;

    static void initializeInstance();
    static void finalizeInstance();
    static Commandline* getInstance();

    void registerCommand(const stl::string &text, ICLCommand *command);
    void unregisterCommand(const stl::string &text);

    void execute(const stl::string &text);
    void pushCommand(const stl::string &text);
    void flush();

private:
    Commandline();
    ~Commandline();

    static Commandline *s_inst;
    istMemberPtrDecl(Members) m;
};

#ifdef ist_enable_Commandline
#   define istCommandlineInitialize()          ist::Commandline::initializeInstance()
#   define istCommandlineFinalize()            ist::Commandline::finalizeInstance()
#   define istCommandlineExecute(S)            ist::Commandline::getInstance()->execute(S)
#   define istCommandlinePush(S)               ist::Commandline::getInstance()->pushCommand(S)
#   define istCommandlineFlush()               ist::Commandline::getInstance()->flush()
#   define istCommandlineRegister(S, ...)      ist::Commandline::getInstance()->registerCommand(S, ist::CreateCLCommand(__VA_ARGS__))
#   define istCommandlineUnregister(S)         ist::Commandline::getInstance()->unregisterCommand(S)
#else // ist_enable_Commandline
#   define istCommandlineInitialize()          
#   define istCommandlineFinalize()            
#   define istCommandlineExecute(S)            
#   define istCommandlinePush(S)               
#   define istCommandlineFlush()               
#   define istCommandlineRegister(S, ...)      
#   define istCommandlineUnregister(S)         
#endif // ist_enable_Commandline

} // namespace ist
#endif // ist_Debug_Commandline_h
