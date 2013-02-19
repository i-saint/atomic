#include "stdafx.h"
#include "GameServer.h"
#include "GameClient.h"
#include "Game/AtomicApplication.h"

namespace atomic {

#ifdef atomic_enable_GameClient

GameClient * GameClient::s_inst;

void GameClient::initializeInstance()
{
    if(!s_inst) {
        s_inst = new GameClient();
    }
}

void GameClient::finalizeInstance()
{
    if(s_inst) {
        delete s_inst;
        s_inst = NULL;
    }
}

GameClient* GameClient::getInstance()
{
    return s_inst;
}



GameClient::GameClient()
    : m_stop(false)
    , m_thread(NULL)
    , m_pid(0)
{
}

GameClient::~GameClient()
{
    shutdown();
}

void GameClient::setEventHandler( const EventHandler &h )
{
    m_handler = h;
}

void GameClient::connect( const char *host, uint16 port )
{
    shutdown();
    m_address = Poco::Net::SocketAddress(host, port);
    m_thread = istNew(ist::FunctorThread<>)( std::bind(&GameClient::messageLoop, this) );
}

void GameClient::close()
{
    m_stop = true;
}

void GameClient::shutdown()
{
    close();
    if(m_thread) {
        m_thread->join();
        delete m_thread;
    }
    m_pid = 0;
    m_stop = false;
}

void GameClient::handleEvent( Event e )
{
    if(m_handler) {
        m_handler(this, e);
    }
}

void GameClient::messageLoop()
{
    ist::Thread::setNameToCurrentThread("GameClient::messageLoop()");
    {
        ist::Mutex::ScopedLock slock(m_mutex_send);
        m_message_send.insert(m_message_send.begin(), PMessage_Join::create(0, atomicGetConfig()->name));
    }

    Poco::Net::StreamSocket *sock = NULL;
    Poco::Net::SocketStream *stream = NULL;
    try {
        sock = new Poco::Net::StreamSocket(m_address);
        sock->setNoDelay(true);
        sock->setBlocking(true);
        sock->setReceiveTimeout(Poco::Timespan(atomic_NetworkTimeout, 0));
        sock->setSendTimeout(Poco::Timespan(atomic_NetworkTimeout, 0));
        stream = new Poco::Net::SocketStream(*sock);
    }
    catch(Poco::Exception &) {
        handleEvent(EV_ConnectionFailed);
        goto Cleanup;
    }
    handleEvent(EV_Connected);

    while(!m_stop) {
        try {
            sendMessage(stream);
            recvMessage(stream);
        }
        catch(Poco::TimeoutException &) {
            m_stop = true;
            handleEvent(EV_Diconnected);
        }
        catch(Poco::Exception &) {
            m_stop = true;
            handleEvent(EV_Diconnected);
        }
    }

    // todo:
    // stream->write(PM_Leave);
    handleEvent(EV_End);

    sock->shutdown();
Cleanup:
    delete stream;
    delete sock;
    m_stop = false;
}

#else // atomic_enable_GameClient
#endif // atomic_enable_GameClient

} // namespace atomic
