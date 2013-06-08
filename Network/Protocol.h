#ifndef atm_Network_PMessage_h
#define atm_Network_PMessage_h
#include "externals.h"
#include "types.h"
#include "Game/Input.h"
#include "LevelEditorCommand.h"

namespace atm {

// パケットのヘッダ
struct PMesBufferHeader
{
    char magic[8];      // "atomic\0\0"
    uint16 version;
    uint16 num_message;
    uint32 length_in_byte;
};


static const char PM_message_header[8] = "atomic\0";


enum PM_Type
{
    PM_Unknown,
    PM_Ping,
    PM_Pong,
    PM_Join,
    PM_Accepted,
    PM_Rejected,
    PM_Leave,
    PM_GameStart,
    PM_GameEnd,
    PM_Update,
    PM_Text,
    PM_Sync,
    PM_LevelEditorCommand,
};

union istAlign(16) PMessage
{
    struct {
        PM_Type type;
    };
    uint8 padding[64];

    PMessage() : type(PM_Unknown) {}
    void share();
    void destroy(); // デストラクタ代わり。可変長系メッセージのメモリの開放はこれで行う
};
atmGlobalNamespace(
    istSerializeRaw(atm::PMessage)
    )

struct PBuffer
{
    int32 ref_count;
    uint32 data_size;

    static PBuffer* construct(void *mem, uint32 size); // size: この構造体自身は含まないサイズ
    static void destruct(void *mem);
    char* getData() { return reinterpret_cast<char*>(this+1); }
    const char* getData() const { return reinterpret_cast<const char*>(this+1); }
};

typedef ist::raw_vector<PMessage> PMessageCont;
typedef ist::raw_vector<char> PMessageBuffer;

bool SendPMessages(Poco::Net::SocketStream *stream, PMessageBuffer &buf, PMessageCont &messages);
bool RecvPMessages(Poco::Net::SocketStream *stream, PMessageBuffer &buf, PMessageCont &messages);
void DestructMessages(PMessageCont &messages);

#define PM_Ensure(T) istStaticAssert(sizeof(T)==sizeof(PMessage))

template<class T>
const PMessage& PMessageCast(const T &mes)
{
    PM_Ensure(T);
    return reinterpret_cast<const PMessage&>(mes);
}


// client -> server
struct istAlign(16) PMessage_Ping
{
    PM_Type type;
    uint8 padding[60];

    static PMessage_Ping create();
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Ping);


// client <- server
struct istAlign(16) PMessage_Pong
{
    PM_Type type;
    uint8 padding[60];

    static PMessage_Pong create();
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Pong);


// client <- server
struct istAlign(16) PMessage_Accepted
{
    PM_Type type;
    PlayerID player_id;
    uint8 padding[56];

    static PMessage_Accepted create(PlayerID pid);
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Accepted);


// client <- server
struct istAlign(16) PMessage_Rejected
{
    PM_Type type;
    uint8 padding[60];

    static PMessage_Rejected create();
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Rejected);


// client -> server
struct istAlign(16) PMessage_Join
{
    PM_Type type;
    PlayerID player_id;
    uint32 frame;
    wchar_t name[16];
    uint8 padding[20];

    static PMessage_Join create(PlayerID pid, const PlayerName &name);
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Join);


// client -> server
// all clients <- server
struct istAlign(16) PMessage_Leave
{
    PM_Type type;
    PlayerID player_id;
    uint8 padding[56];

    static PMessage_Leave create(PlayerID pid);
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Leave);


// all clients <- server
struct istAlign(16) PMessage_GameStart
{
    PM_Type type;
    uint8 padding[60];

    PMessage_GameStart() : type(PM_GameStart) {}
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_GameStart);

// all clients <- server
struct istAlign(16) PMessage_GameEnd
{
    PM_Type type;
    uint8 padding[60];

    PMessage_GameEnd() : type(PM_GameEnd) {}
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_GameEnd);


// client -> server
// all clients <- server
struct istAlign(16) PMessage_Update
{
    PM_Type type;
    PlayerID player_id;
    uint32 frame;
    uint32 ping;
    uint32 server_frame;
    uint32 sync_mark;
    RepInput input;
    uint8 padding[32];

    PMessage_Update() : type(PM_Update) {}
    static PMessage_Update create(PlayerID pid, uint32 frame, const RepInput &inp);
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Update);


// client -> server
// all clients <- server
struct istAlign(16) PMessage_Text
{
    PM_Type type;
    uint32 player_id;
    wchar_t text[28];

    PMessage_Text() : type(PM_Text) {}
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Text);


// client <- server
// client -> server
struct istAlign(16) PMessage_Sync
{
    PM_Type type;
    uint32 data_size;
    void *data;
    uint8 padding[52];

    PMessage_Sync() : type(PM_Sync), data_size(0), data(NULL) {}
    operator const PMessage&() const { return reinterpret_cast<const PMessage&>(*this); }
};
PM_Ensure(PMessage_Sync);

// client -> server
// all clients <- server
struct istAlign(16) PMessage_LEC
{
    PM_Type type;
    LevelEditorCommand lec;
    uint8 padding[16];

    static PMessage_LEC create(const LevelEditorCommand &lec);
};
PM_Ensure(PMessage_LEC);

#undef PM_Ensure



class PMessenger
{
public:
    typedef std::function<void (const PMessage &)> MessageHandler;
    typedef std::function<void (const PMessageCont &)> MessageContHandler;

    virtual ~PMessenger();
    void pushMessage(const PMessage &p);
    void pushMessage(const PMessageCont &p);
    void pushMessage(const PMessage *p, size_t num);
    void handleReceivedMessage(const MessageHandler &h);
    void handleReceivedMessageCont(const MessageContHandler &h);

protected:
    virtual void processReceivingMessage(PMessageCont &mes) {}

    bool sendMessage(Poco::Net::SocketStream *stream);
    bool recvMessage(Poco::Net::SocketStream *stream);
    void clearAllMessage();

protected:
    PMessageBuffer m_message_buffer;

    ist::Mutex m_mutex_send;
    PMessageCont m_message_send;
    PMessageCont m_message_sending;

    ist::Mutex m_mutex_recv;
    PMessageCont m_message_recv;
    PMessageCont m_message_receiving;
    PMessageCont m_message_consuming;
};

} // namespace atm

#endif // atm_Network_PMessage_h
