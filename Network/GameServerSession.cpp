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
    ss.setReceiveTimeout(Poco::Timespan(atomic_NetworkTimeout, 0));
    ss.setSendTimeout(Poco::Timespan(atomic_NetworkTimeout, 0));
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
    {
        ist::Mutex::ScopedLock slock(m_mutex_send);
        m_message_send.insert(m_message_send.begin(), PMessage_Accepted::create(m_pid));
    }

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

void GameServerSession::processReceivingMessage( PMessageCont &cont )
{
    for(size_t i=0; i<cont.size(); ++i) {
        PMessage &mes = cont[i];
        switch(mes.type) {
        case PM_Join:
            {
                reinterpret_cast<PMessage_Join&>(mes).player_id = m_pid;
            }
            break;
        case PM_Update:
            {
                reinterpret_cast<PMessage_Update&>(mes).player_id = m_pid;
            }
            break;
        }

    }
}


Poco::Net::TCPServerConnection* GameServerSessionFactory::createConnection( const Poco::Net::StreamSocket &ss )
{
    return new GameServerSession(ss);
}


#endif // atomic_enable_GameServer

} // namespace atomic
