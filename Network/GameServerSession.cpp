#include "stdafx.h"
#include "GameServer.h"
#include "GameServerSession.h"

namespace atomic {

class GameServerSession : public Poco::Net::TCPServerConnection
{
typedef Poco::Net::TCPServerConnection super;
public:
    GameServerSession(const Poco::Net::StreamSocket &ss);
    virtual void run();

private:
};

GameServerSession::GameServerSession( const Poco::Net::StreamSocket &_ss )
    : super(_ss)
{
    Poco::Net::StreamSocket &ss = socket();
    ss.setNoDelay(true);
    ss.setReceiveTimeout(Poco::Timespan(3, 0));
    ss.setSendTimeout(Poco::Timespan(3, 0));
}

void GameServerSession::run()
{

}


Poco::Net::TCPServerConnection* GameServerSessionFactory::createConnection( const Poco::Net::StreamSocket &ss )
{
    return new GameServerSession(ss);
}

} // namespace atomic
