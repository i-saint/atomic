#ifndef atomic_Network_GameClientSession_h
#define atomic_Network_GameClientSession_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {


class GameClientSessionFactory : public Poco::Net::TCPServerConnectionFactory
{
public:
    virtual Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket &ss);
};

} // namespace atomic
#endif // atomic_Network_GameClientSession_h
