#ifndef atomic_Network_GameServer_h
#define atomic_Network_GameServer_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {

#ifdef atomic_enable_GameServer

class GameServer
{
public:
    enum ErrorCode {
        ER_Ok,
        ER_StartFailed,
    };

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

#define atomicGameServerInitialize()    GameServer::initializeInstance()
#define atomicGameServerFinalize()      GameServer::finalizeInstance()

#else // atomic_enable_GameServer

#define atomicGameServerInitialize()    
#define atomicGameServerFinalize()      

#endif // atomic_enable_GameServer

} // namespace atomic
#endif // atomic_Network_GameServer_h
