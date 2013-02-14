#include "stdafx.h"
#include "GameServer.h"
#include "GameServerSession.h"

namespace atomic {

#ifdef atomic_enable_GameServer

GameServer* GameServer::s_inst;

void GameServer::initializeInstance()
{
    if(!s_inst) {
        s_inst = new GameServer();
        s_inst->start();
    }
}

void GameServer::finalizeInstance()
{
    if(s_inst) {
        delete s_inst;
        s_inst = NULL;
    }
}

GameServer* GameServer::getInstance()
{
    return s_inst;
}


GameServer::GameServer()
    : m_server(NULL)
    , m_message_thread(NULL)
    , m_pidgen(0)
{
}

GameServer::~GameServer()
{
    stop();
}

void GameServer::start()
{
    if(!m_server) {
        Poco::Net::TCPServerParams* params = new Poco::Net::TCPServerParams;
        params->setMaxQueued(10);
        params->setMaxThreads(8);
        params->setThreadIdleTime(Poco::Timespan(3, 0));

        try {
            Poco::Net::ServerSocket svs(atomic_GameServer_DefaultPort);
            m_server = new Poco::Net::TCPServer(new GameServerSessionFactory(), svs, params);
            m_server->start();

            m_message_thread = istNew(ist::FunctorThread<>)(std::bind(&GameServer::messageLoop, this));
        }
        catch(Poco::IOException &e) {
            istAssert(e.what());
            stop();
        }
    }
}

void GameServer::stop()
{
    if(m_server) {
        m_server->stop();
        while(m_server->currentConnections()>0 || m_server->currentThreads()>0) {
            ist::Thread::milliSleep(5);
        }
        delete m_server;
        m_server = NULL;
    }
    if(m_message_thread) {
        m_message_thread->join();
        delete m_message_thread;
        m_message_thread = NULL;
    }
}

void GameServer::restart()
{
    stop();
    start();
}

void GameServer::handleMessageCont( const PMessageCont &cont )
{
    m_mes_recved.insert(m_mes_recved.end(), cont.begin(), cont.end());
}

void GameServer::addSession( GameServerSession *s )
{
    ist::Mutex::ScopedLock l(m_mtx_sessions);
    m_sessions.push_back(s);
}

void GameServer::eraseSession( GameServerSession *s )
{
    ist::Mutex::ScopedLock l(m_mtx_sessions);
    m_sessions.erase( std::find(m_sessions.begin(), m_sessions.end(), s) );
}

void GameServer::pushMessage( PMessage &mes )
{
    ist::Mutex::ScopedLock l(m_mtx_messages);
    m_mes_pushed.push_back(mes);
}

void GameServer::recvMessage()
{
    {
        ist::Mutex::ScopedLock l(m_mtx_sessions);
        for(size_t i=0; i<m_sessions.size(); ++i) {
            m_sessions[i]->handleReceivedMessageCont(m_receiver);
        }
    }
    {
        ist::Mutex::ScopedLock l(m_mtx_messages);
        m_mes_recved.insert(m_mes_recved.end(), m_mes_pushed.begin(), m_mes_pushed.end());
        m_mes_pushed.clear();
    }

    for(size_t i=0; i<m_mes_recved.size(); ++i) {
        m_mes_send.push_back(m_mes_recved[i]);
    }
}

void GameServer::sendMessage()
{
    {
        ist::Mutex::ScopedLock l(m_mtx_sessions);
        for(size_t i=0; i<m_sessions.size(); ++i) {
            m_sessions[i]->pushMessage(m_mes_send);
        }
    }
    DestructMessages(m_mes_send);
}

void GameServer::messageLoop()
{
    ist::Thread::setNameToCurrentThread("GameServer::messageLoop()");
    m_receiver = std::bind(&GameServer::handleMessageCont, this, std::placeholders::_1);

    ist::Timer timer;
    while(m_server!=NULL) {
        timer.reset();
        recvMessage();
        sendMessage();
        if(timer.getElapsedMillisec() < 10.0f) {
            ist::Thread::milliSleep(5);
        }
    }
}

uint32 GameServer::cretaePID()
{
    return m_pidgen++;
}

#endif // atomic_enable_GameServer

} // namespace atomic
