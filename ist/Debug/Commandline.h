#ifndef ist_Debug_Commandline_h
#define ist_Debug_Commandline_h
#include "ist/Base/Types.h"
#include "CommandlineCommand.h"

namespace ist {

class istInterModule Commandline
{
public:
    typedef stl::map<std::string, ICLCommand*> CommandCont;

    static void initializeInstance();
    static void finalizeInstance();
    static Commandline* getInstance();

    void registerCommand(const stl::string &text, ICLCommand *command);
    void unregisterCommand(const stl::string &text);
    void execute(const stl::string &text);

private:
    Commandline();
    ~Commandline();

    static Commandline *s_inst;
    CommandCont m_commands;
};


} // namespace ist
#endif // ist_Debug_Commandline_h
