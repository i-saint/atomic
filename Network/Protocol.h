#ifndef atomic_Network_PMessage_h
#define atomic_Network_PMessage_h
#include "externals.h"
#include "Game/Input.h"

namespace atomic {

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
};

union istAlign(16) PMessage
{
    struct {
        PM_Type type;
    };
    uint8 padding[64];

    PMessage() : type(PM_Unknown) {}
    void destroy(); // デストラクタ代わり。可変長系メッセージのメモリの開放はこれで行う
};
typedef ist::raw_vector<PMessage> PMessageCont;
typedef ist::raw_vector<char> PMessageBuffer;

bool SendPMessages(Poco::Net::StreamSocket *stream, PMessageBuffer &buf, PMessageCont &messages);
bool RecvPMessages(Poco::Net::StreamSocket *stream, PMessageBuffer &buf, PMessageCont &messages);
void DestructMessages(PMessageCont &messages);

#define PM_Ensure(T) BOOST_STATIC_ASSERT(sizeof(T)==sizeof(PMessage))

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

    PMessage_Ping() : type(PM_Ping) {}
};
PM_Ensure(PMessage_Ping);


// client <- server
struct istAlign(16) PMessage_Pong
{
    PM_Type type;
    uint8 padding[60];

    PMessage_Pong() : type(PM_Pong) {}
};
PM_Ensure(PMessage_Pong);



// client -> server
struct istAlign(16) PMessage_Join
{
    PM_Type type;
    uint32 equip;
    wchar_t name[16];
    uint8 padding[24];

    PMessage_Join() : type(PM_Join) {}
};
PM_Ensure(PMessage_Join);


// client <- server
struct istAlign(16) PMessage_Accepted
{
    PM_Type type;
    uint32 player_id;
    uint8 padding[56];

    PMessage_Accepted() : type(PM_Accepted), player_id(0) {}
};
PM_Ensure(PMessage_Accepted);


// client <- server
struct istAlign(16) PMessage_Rejected
{
    PM_Type type;
    uint8 padding[60];

    PMessage_Rejected() : type(PM_Rejected) {}
};
PM_Ensure(PMessage_Rejected);


// client -> server
// all clients <- server
struct istAlign(16) PMessage_Leave
{
    PM_Type type;
    uint8 padding[60];

    PMessage_Leave() : type(PM_Leave) {}
};
PM_Ensure(PMessage_Leave);


// all clients <- server
struct istAlign(16) PMessage_GameStart
{
    PM_Type type;
    uint8 padding[60];

    PMessage_GameStart() : type(PM_GameStart) {}
};
PM_Ensure(PMessage_GameStart);

// all clients <- server
struct istAlign(16) PMessage_GameEnd
{
    PM_Type type;
    uint8 padding[60];

    PMessage_GameEnd() : type(PM_GameEnd) {}
};
PM_Ensure(PMessage_GameEnd);


// client -> server
// all clients <- server
struct istAlign(16) PMessage_Update
{
    PM_Type type;
    uint32 player_id;
    uint32 frame;
    uint32 sync_mark;
    RepInput input;
    uint8 padding[40];

    PMessage_Update() : type(PM_Update) {}
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
};
PM_Ensure(PMessage_Sync);

#undef PM_Ensure



class PMessenger
{
public:
    typedef std::function<void (const PMessage &)> MessageHandler;

    virtual ~PMessenger();
    void pushMessage(const PMessage &p);
    void pushMessage(const PMessage *p, size_t num);
    void handleReceivedMessage(const MessageHandler &h);

protected:
    bool sendMessage(Poco::Net::StreamSocket *stream);
    bool recvMessage(Poco::Net::StreamSocket *stream);
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

} // namespace atomic

#endif // atomic_Network_PMessage_h
