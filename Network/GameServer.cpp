#include "stdafx.h"
#include "GameServer.h"
#include "GameSession.h"

namespace atomic {




class GameSessionFactory : public Poco::Net::TCPServerConnectionFactory
{
public:
    virtual Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& socket)
    {
        return new GameSession(socket);
    }
};


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
            Poco::Net::ServerSocket svs(10051);
            m_server = new Poco::Net::TCPServer(new GameSessionFactory(), svs, params);
            m_server->start();
        }
        catch(Poco::IOException &e) {
            istAssert(e.what());
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
}

void GameServer::restart()
{
    stop();
    start();
}

} // namespace atomic
