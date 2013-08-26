#ifndef atm_Engine_Network_GameServer_h
#define atm_Engine_Network_GameServer_h
#include "externals.h"
#include "Protocol.h"

namespace atm {

#ifdef atm_enable_GameServer

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
    bool getStopFlag() const { return m_stop; }

private:
    typedef ist::vector<GameServerSession*> SessionCont;
    static GameServer *s_inst;
    Poco::Net::TCPServer *m_server;
    ist::Thread *m_message_thread;
    atomic_int32 m_pidgen;
    uint32 m_frame;
    uint32 m_delay;
    bool m_stop;

    ist::Mutex m_mtx_sessions;
    SessionCont m_sessions;

    ist::Mutex m_mtx_messages;
    PMessenger::MessageContHandler m_receiver;
    PMessageCont m_mes_pushed;
    PMessageCont m_mes_recved;
    PMessageCont m_mes_send;
};

#define atmGameServerInitialize()    GameServer::initializeInstance()
#define atmGameServerFinalize()      GameServer::finalizeInstance()
#define atmGameServerGet()           GameServer::getInstance()

#else // atm_enable_GameServer

#define atmGameServerInitialize()    
#define atmGameServerFinalize()      
#define atmGameServerGet()           

#endif // atm_enable_GameServer

} // namespace atm
#endif // atm_Engine_Network_GameServer_h
