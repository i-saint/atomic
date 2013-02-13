#include "istPCH.h"
#include "Commandline.h"
#include "CommandlineCommand.h"

namespace ist {

Commandline * Commandline::s_inst;

void Commandline::initializeInstance()
{
    if(s_inst==NULL) {
        s_inst = istNew(Commandline)();
    }
}

void Commandline::finalizeInstance()
{
    if(s_inst!=NULL) {
        istSafeDelete(s_inst);
    }
}

Commandline* Commandline::getInstance()
{
    return s_inst;
}


Commandline::Commandline()
{
}

Commandline::~Commandline()
{
    while(!m_commands.empty()) {
        unregisterCommand(m_commands.begin()->first);
    }
}

void Commandline::registerCommand( const stl::string &text, ICLCommand *command )
{
    ICLCommand *&cmd = m_commands[text];
    if(cmd) { cmd->release(); }
    cmd = command;
}

void Commandline::unregisterCommand( const stl::string &text )
{
    CommandCont::iterator p = m_commands.find(text);
    if(p!=m_commands.end()) {
        p->second->release();
        m_commands.erase(p);
    }
}

void Commandline::execute( const stl::string &text )
{
    stl::string::const_iterator tok = std::find(text.begin(), text.end(), ' ');
    CommandCont::iterator cmdp = m_commands.find(stl::string(text.begin(), tok));
    if(cmdp==m_commands.end()) {
        return;
    }

    ICLCommand *cmd = cmdp->second;
    for(uint32 i=0; i<cmd->getNumArgs(); ++i) {
        if(tok==text.end()) { break; }
        cmd->setArg(i, &(*tok));
    }
    cmd->exec();
}


} // namespace ist
