#ifndef atomic_Network_GameSession_h
#define atomic_Network_GameSession_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {

class GameSession : public Poco::Net::TCPServerConnection
{
typedef Poco::Net::TCPServerConnection super;
public:
    GameSession(const Poco::Net::StreamSocket& s);
    virtual void run();

private:
};

} // namespace atomic
#endif // atomic_Network_GameSession_h
