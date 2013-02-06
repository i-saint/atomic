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

void GameClient::pushCommand( const Protocol &p )
{
    ist::Mutex::ScopedLock lock(m_mutex);
    m_message.push_back(p);
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
    Poco::Net::SocketStream *stream = NULL;
    try {
        sock = new Poco::Net::StreamSocket(m_address);
        sock->setNoDelay(true);
        sock->setBlocking(true);
        sock->setReceiveTimeout(Poco::Timespan(3, 0));
        sock->setSendTimeout(Poco::Timespan(3, 0));

        // todo:
        // greeting
        // stream->write();
        // stream->read();
    }
    catch(Poco::Exception &) {
        handleEvent(EV_ConnectionFailed);
        goto Cleanup;
    }
    handleEvent(EV_Connected);

    stream = new Poco::Net::SocketStream(*sock);
    while(!m_end_flag) {
        size_t received = 0;
        try {
            // todo:
            // stream->write();
            // stream->read();
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
    // stream->write(PTCL_Leave);
    handleEvent(EV_End);

    sock->shutdown();
Cleanup:
    delete stream;
    delete sock;
    m_end_flag = false;
}

} // namespace atomic
