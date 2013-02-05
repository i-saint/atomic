#ifndef atomic_Network_GameServerSession_h
#define atomic_Network_GameServerSession_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {


class GameServerSessionFactory : public Poco::Net::TCPServerConnectionFactory
{
public:
    virtual Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket &ss);
};


} // namespace atomic
#endif // atomic_Network_GameServerSession_h
