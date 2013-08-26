#include "atmPCH.h"
#include "GameServer.h"
#include "GameServerSession.h"

namespace atm {

#ifdef atm_enable_GameServer

GameServerSession::GameServerSession( const Poco::Net::StreamSocket &_ss )
    : super(_ss)
    , m_pid(0)
    , m_frame(0)
    , m_ping(0)
{
    Poco::Net::StreamSocket &ss = socket();
    ss.setNoDelay(true);
    ss.setBlocking(true);
    ss.setReceiveTimeout(Poco::Timespan(atm_NetworkTimeout, 0));
    ss.setSendTimeout(Poco::Timespan(atm_NetworkTimeout, 0));
}

void GameServerSession::run()
{
    atmGameServerGet()->addSession(this);
    try {
        messageLoop();
    }
    catch(...) {
    }
    atmGameServerGet()->eraseSession(this);
}

void GameServerSession::messageLoop()
{
    ist::Thread::setNameToCurrentThread("GameServerSession::messageLoop()");
    m_pid = atmGameServerGet()->cretaePID();
    {
        ist::Mutex::ScopedLock slock(m_mutex_send);
        m_message_send.insert(m_message_send.begin(), PMessage_Accepted::create(m_pid));
    }

    Poco::Net::SocketStream stream(socket());
    ist::Timer timer;
    while(!atmGameServerGet()->getStopFlag()) {
        timer.reset();

        recvMessage(&stream);
        sendMessage(&stream);

        m_ping = uint32(timer.getElapsedMillisec());
        if(timer.getElapsedMillisec()<1.0f) { ist::MiliSleep(1); }
   }
}

void GameServerSession::processReceivingMessage( PMessageCont &cont )
{
    for(size_t i=0; i<cont.size(); ++i) {
        PMessage &mes = cont[i];
        switch(mes.type) {
        case PM_Join:
            {
                auto &m = reinterpret_cast<PMessage_Join&>(mes);
                m.player_id = m_pid;
            }
            break;
        case PM_Update:
            {
                auto &m = reinterpret_cast<PMessage_Update&>(mes);
                m.player_id = m_pid;
                m.ping = m_ping;
                m_frame = m.frame;
            }
            break;
        case PM_Leave:
            {
                auto &m = reinterpret_cast<PMessage_Leave&>(mes);
                m.player_id = m_pid;
            }
            break;
        }
    }
}

uint32 GameServerSession::getFrame() const
{
    return m_frame;
}

uint32 GameServerSession::getAveragePing() const
{
    // todo
    return m_ping;
}


Poco::Net::TCPServerConnection* GameServerSessionFactory::createConnection( const Poco::Net::StreamSocket &ss )
{
    return new GameServerSession(ss);
}


#endif // atm_enable_GameServer

} // namespace atm
