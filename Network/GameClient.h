#ifndef atm_Network_GameClient_h
#define atm_Network_GameClient_h
#include "externals.h"
#include "Protocol.h"

namespace atm {

#ifdef atm_enable_GameClient

class GameClient : public PMessenger
{
typedef PMessenger super;
public:
    struct ClientStates
    {
        PlayerName name;
        PlayerID pid;
        uint32 ping;

        ClientStates() { istMemset(this, 0, sizeof(*this)); }
    };
    typedef stl::map<PlayerID, ClientStates> ClientStatesCont;

    enum Event {
        EV_Unknown,
        EV_Connected,       // 接続した時
        EV_ConnectionFailed,// 接続失敗したとき
        EV_Diconnected,     // ネットワークかサーバーの異常で切断されたとき (正常切断時は End)
        EV_End,             // 切断したとき
    };
    typedef std::function<void (GameClient*, Event)> EventHandler;
    using super::MessageHandler;

    static void initializeInstance();
    static void finalizeInstance();
    static GameClient* getInstance();

    GameClient();
    ~GameClient();

    void setEventHandler(const EventHandler &h);

    // non-blocking
    // エラーの際は setEventHandler() で事前に登録したイベントハンドラに対応するイベントが飛ぶ。
    // connect() は既に接続がある場合それを切断 (ブロッキング処理) してから接続を試みる
    void connect(const char *host, uint16 port);

    // non-blocking
    // 終了リクエストを出す
    // 実際に接続が閉じた時はイベントハンドラにイベントが飛ぶ
    void close();

    using super::sendMessage;
    using super::handleReceivedMessage;

    PlayerID getPlayerID() const { return m_pid; }

    ClientStatesCont& getClientStates() { return m_client_states; }

private:
    void processReceivingMessage(PMessageCont &cont);
    void shutdown();
    void handleEvent(Event e);
    void messageLoop();

private:
    static GameClient *s_inst;

    bool m_stop;
    PlayerID m_pid;
    Poco::Net::SocketAddress m_address;
    EventHandler m_handler;
    ist::Thread *m_thread;
    ClientStatesCont m_client_states;
};


#define atmGameClientInitialize()        GameClient::initializeInstance()
#define atmGameClientFinalize()          GameClient::finalizeInstance()
#define atmGameClientGet()               GameClient::getInstance()
#define atmGameClientConnect(Host,Port)  atmGameClientGet()->connect(Host,Port)
#define atmGameClientPushMessage(m)      atmGameClientGet()->pushMessage(PMessageCast(m))
#define atmGameClientHandleMessages(h)   atmGameClientGet()->handleReceivedMessage(h)
#define atmGameClientGetPlayerID()       atmGameClientGet()->getPlayerID()

#else // atm_enable_GameClient

#define atmGameClientInitialize()        
#define atmGameClientFinalize()          
#define atmGameClientGet()               
#define atmGameClientConnect(Host,Port)  
#define atmGameClientPushMessage(m)      
#define atmGameClientHandleMessages(h)   
#define atmGameClientGetPlayerID()       

#endif // atm_enable_GameClient

} // namespace atm
#endif // atm_Network_GameClient_h
