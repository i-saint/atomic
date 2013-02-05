#ifndef atomic_Network_GameServer_h
#define atomic_Network_GameServer_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {

class GameServer
{
public:
    static void initializeInstance();
    static void finalizeInstance();
    static GameServer* getInstance();

    void start();
    void stop();
    void restart();

private:
    GameServer();
    ~GameServer();

private:
    static GameServer *s_inst;
    Poco::Net::TCPServer *m_server;
};

} // namespace atomic
#endif // atomic_Network_GameServer_h
