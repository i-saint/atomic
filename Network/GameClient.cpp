#include "stdafx.h"
#include "GameServer.h"
#include "GameClient.h"

namespace atomic {

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
    : m_end_flag(false)
    , m_thread(NULL)
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
    m_thread = istNew(ist::FunctorThread<>)( std::bind(&GameClient::networkLoop, this) );
}

void GameClient::close()
{
    m_end_flag = true;
}

void GameClient::sendMessage( const PMessage &p )
{
    ist::Mutex::ScopedLock lock(m_mutex_send);
    m_message_send.push_back(p);
}

void GameClient::handleReceivedMessage( const MessageHandler &h )
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_recv);
        m_message_consuming = m_message_recv;
        m_message_recv.clear();
    }
    for(size_t i=0; i<m_message_consuming.size(); ++i) {
        h(m_message_consuming[i]);
    }
    DestructMessages(m_message_consuming);
}

bool GameClient::sendMessage(Poco::Net::StreamSocket *stream)
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_send);
        m_message_sending = m_message_send;
        m_message_send.clear();
    }
    bool ret = SendPMessages(stream, m_message_buffer, m_message_sending);
    DestructMessages(m_message_sending);
    return ret;
}

bool GameClient::recvMessage(Poco::Net::StreamSocket *stream)
{
    bool ret = RecvPMessages(stream, m_message_buffer, m_message_receiving);
    {
        ist::Mutex::ScopedLock lock(m_mutex_recv);
        m_message_recv.insert(m_message_recv.end(), m_message_receiving.begin(), m_message_receiving.end());
    }
    m_message_receiving.clear();
    return ret;
}



void GameClient::shutdown()
{
    close();
    if(m_thread) {
        m_thread->join();
        delete m_thread;
    }
}

void GameClient::handleEvent( Event e )
{
    if(m_handler) {
        m_handler(this, e);
    }
}

void GameClient::networkLoop()
{
    Poco::Net::StreamSocket *sock = NULL;
    try {
        sock = new Poco::Net::StreamSocket(m_address);
        sock->setNoDelay(true);
        sock->setBlocking(true);
        sock->setReceiveTimeout(Poco::Timespan(3, 0));
        sock->setSendTimeout(Poco::Timespan(3, 0));
        {
            // todo:
            // greeting
            ist::Mutex::ScopedLock slock(m_mutex_send);
            ist::Mutex::ScopedLock rlock(m_mutex_recv);
            m_message_send.clear();
            m_message_recv.clear();
            sendMessage(sock);
            recvMessage(sock);
        }
    }
    catch(Poco::Exception &) {
        handleEvent(EV_ConnectionFailed);
        goto Cleanup;
    }
    handleEvent(EV_Connected);

    while(!m_end_flag) {
        size_t received = 0;
        try {
            sendMessage(sock);
            recvMessage(sock);
        }
        catch(Poco::Exception &) {
            // おそらく connection time out
        }
        if(received==0) {
            handleEvent(EV_Diconnected);
            goto Cleanup;
        }
    }

    // todo:
    // stream->write(PM_Leave);
    handleEvent(EV_End);

    sock->shutdown();
Cleanup:
    delete sock;
    m_end_flag = false;
}

} // namespace atomic
