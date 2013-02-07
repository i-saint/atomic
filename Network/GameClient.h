#ifndef atomic_Network_GameClient_h
#define atomic_Network_GameClient_h
#include "externals.h"
#include "Protocol.h"

namespace atomic {



class GameClient : public PMessenger
{
typedef PMessenger super;
public:
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

private:
    void shutdown();
    void handleEvent(Event e);
    void networkLoop();

private:
    static GameClient *s_inst;

    bool m_end_flag;
    Poco::Net::SocketAddress m_address;
    EventHandler m_handler;
    ist::Thread *m_thread;
};

} // namespace atomic
#endif // atomic_Network_GameClient_h
