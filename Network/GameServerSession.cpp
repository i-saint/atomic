#include "stdafx.h"
#include "GameServer.h"
#include "GameServerSession.h"

namespace atomic {

#ifdef atomic_enable_GameServer

GameServerSession::GameServerSession( const Poco::Net::StreamSocket &_ss )
    : super(_ss)
    , m_pid(0)
{
    Poco::Net::StreamSocket &ss = socket();
    ss.setNoDelay(true);
    ss.setBlocking(true);
    ss.setReceiveTimeout(Poco::Timespan(3, 0));
    ss.setSendTimeout(Poco::Timespan(3, 0));
}

void GameServerSession::run()
{
    atomicGameServerGet()->addSession(this);
    try {
        messageLoop();
    }
    catch(...) {
    }
    atomicGameServerGet()->eraseSession(this);
}

void GameServerSession::messageLoop()
{
    ist::Thread::setNameToCurrentThread("GameServerSession::messageLoop()");
    m_pid = atomicGameServerGet()->cretaePID();

    Poco::Net::StreamSocket *stream = &socket();
    ist::Timer timer;
    for(;;) {
        timer.reset();

        recvMessage(stream);
        sendMessage(stream);

        if(timer.getElapsedMillisec()<1.0f) {
            ist::Thread::milliSleep(1);
        }
   }
}


Poco::Net::TCPServerConnection* GameServerSessionFactory::createConnection( const Poco::Net::StreamSocket &ss )
{
    return new GameServerSession(ss);
}


#endif // atomic_enable_GameServer

} // namespace atomic
