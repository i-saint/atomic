#include "stdafx.h"
#include "GameServer.h"
#include "GameServerSession.h"

namespace atomic {

class GameServerSession
    : public Poco::Net::TCPServerConnection
    , public PMessenger
{
typedef Poco::Net::TCPServerConnection super;
public:
    GameServerSession(const Poco::Net::StreamSocket &ss);
    virtual void run();
    void messageLoop();

};

GameServerSession::GameServerSession( const Poco::Net::StreamSocket &_ss )
    : super(_ss)
{
    Poco::Net::StreamSocket &ss = socket();
    ss.setNoDelay(true);
    ss.setBlocking(true);
    ss.setReceiveTimeout(Poco::Timespan(3, 0));
    ss.setSendTimeout(Poco::Timespan(3, 0));
}

void GameServerSession::run()
{
    try {
        messageLoop();
    }
    catch(...) {
    }
}

void GameServerSession::messageLoop()
{
    Poco::Net::StreamSocket *stream = &socket();
    for(;;) {
        recvMessage(stream);
        sendMessage(stream);
   }
}


Poco::Net::TCPServerConnection* GameServerSessionFactory::createConnection( const Poco::Net::StreamSocket &ss )
{
    return new GameServerSession(ss);
}

} // namespace atomic
