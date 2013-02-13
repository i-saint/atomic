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
        while(tok!=text.end() && *tok==' ') { ++tok; }
        if(tok==text.end()) { break; }
        cmd->setArg(i, &(*tok));
        tok = std::find(tok, text.end(), ' ');
    }
    cmd->exec();
}

void Commandline::pushCommand( const stl::string &text )
{
    ist::Mutex::ScopedLock l(m_mutex);
    m_queue.push_back(text);
}

void Commandline::flush()
{
    ist::Mutex::ScopedLock l(m_mutex);
    for(size_t i=0; i<m_queue.size(); ++i) {
        execute(m_queue[i]);
    }
    m_queue.clear();
}


} // namespace ist
