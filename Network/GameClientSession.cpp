#include "stdafx.h"
#include "GameServer.h"
#include "GameClientSession.h"

namespace atomic {

class GameClientSession : public Poco::Net::TCPServerConnection
{
    typedef Poco::Net::TCPServerConnection super;
public:
    GameClientSession(const Poco::Net::StreamSocket& ss);
    virtual void run();

private:
};

GameClientSession::GameClientSession( const Poco::Net::StreamSocket& _ss )
    : super(_ss)
{
    Poco::Net::StreamSocket &ss = socket();
    ss.setNoDelay(true);
    ss.setReceiveTimeout(Poco::Timespan(3, 0));
    ss.setSendTimeout(Poco::Timespan(3, 0));
}

void GameClientSession::run()
{

}


Poco::Net::TCPServerConnection* GameClientSessionFactory::createConnection( const Poco::Net::StreamSocket& ss )
{
    return new GameClientSession(ss);
}

} // namespace atomic
