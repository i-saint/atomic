#ifndef atomic_Network_GameClient_h
#define atomic_Network_GameClient_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {

class GameClient
{
public:
    enum Event {
        EV_Unknown,
        EV_Connected,       // 接続した時
        EV_ConnectionFailed,// 接続失敗したとき
        EV_Diconnected,     // ネットワークかサーバーの異常で切断されたとき (正常切断時は End)
        EV_End,             // 切断したとき
    };
    typedef std::function<void (GameClient*, Event)> EventHandler;
    typedef std::function<void (const PMessage &)> MessageHandler;
    typedef ist::raw_vector<PMessage> MessageCont;
    typedef ist::raw_vector<char> MessageBuffer;

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

    void sendMessage(const PMessage &p);
    void handleReceivedMessage(const MessageHandler &h);

private:
    void shutdown();
    void handleEvent(Event e);
    void sendMessage();
    void recvMessage();
    void networkLoop();

private:
    static GameClient *s_inst;

    bool m_end_flag;
    Poco::Net::SocketAddress m_address;
    EventHandler m_handler;
    ist::Thread *m_thread;

    ist::Mutex m_mutex_send;
    MessageCont m_message_send;
    MessageCont m_message_send_tmp;
    MessageBuffer m_buffer_send;

    ist::Mutex m_mutex_recv;
    MessageCont m_message_recv;
    MessageCont m_message_recv_tmp;
    MessageBuffer m_buffer_recv;
};

} // namespace atomic
#endif // atomic_Network_GameClient_h
