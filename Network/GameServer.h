#ifndef atomic_Network_GameServer_h
#define atomic_Network_GameServer_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {

#ifdef atomic_enable_GameServer

class GameServerSession;

class GameServer
{
friend class GameServerSession;
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
    void handleMessageCont(const PMessageCont &cont);
    void addSession(GameServerSession *s);
    void eraseSession(GameServerSession *s);
    void pushMessage(PMessage &mes);
    void recvMessage();
    void sendMessage();

    void messageLoop();

    uint32 cretaePID();

private:
    typedef ist::vector<GameServerSession*> SessionCont;
    static GameServer *s_inst;
    Poco::Net::TCPServer *m_server;
    ist::Thread *m_message_thread;
    atomic_int32 m_pidgen;

    ist::Mutex m_mtx_sessions;
    SessionCont m_sessions;

    ist::Mutex m_mtx_messages;
    PMessenger::MessageContHandler m_receiver;
    PMessageCont m_mes_pushed;
    PMessageCont m_mes_recved;
    PMessageCont m_mes_send;
};

#define atomicGameServerInitialize()    GameServer::initializeInstance()
#define atomicGameServerFinalize()      GameServer::finalizeInstance()
#define atomicGameServerGet()           GameServer::getInstance()

#else // atomic_enable_GameServer

#define atomicGameServerInitialize()    
#define atomicGameServerFinalize()      
#define atomicGameServerGet()           

#endif // atomic_enable_GameServer

} // namespace atomic
#endif // atomic_Network_GameServer_h
