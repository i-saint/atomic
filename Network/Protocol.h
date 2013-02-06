#ifndef atomic_Network_PMessage_h
#define atomic_Network_PMessage_h
#include "externals.h"

namespace atomic {

// パケットは以下の構造とする
// 
// char[8] magic;           // "atomic\0\0"
// uint32 length_in_byte;
// uint32 num_message;
// PMessage messages[num_message];
// (可変長メッセージの場合ここに追加データ)


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
    PM_Update,
    PM_Text,
    PM_Sync,
};

union istAlign(16) PMessage
{
    struct {
        PM_Type type;
    };
    uint32 dummy[8];

    PMessage() : type(PM_Unknown) {}
    void destroy(); // デストラクタ代わり。可変長系メッセージのメモリの開放はこれで行う
};
#define PM_Ensure(T) BOOST_STATIC_ASSERT(sizeof(T)<=sizeof(PMessage))


struct istAlign(16) PMessage_Ping
{
    PM_Type type;

    PMessage_Ping() : type(PM_Ping) {}
};
PM_Ensure(PMessage_Ping);


struct istAlign(16) PMessage_Pong
{
    PM_Type type;

    PMessage_Pong() : type(PM_Pong) {}
};
PM_Ensure(PMessage_Pong);


struct istAlign(16) PMessage_Join
{
    PM_Type type;

    PMessage_Join() : type(PM_Join) {}
};
PM_Ensure(PMessage_Join);


struct istAlign(16) PMessage_Accepted
{
    PM_Type type;
    uint32 uid;

    PMessage_Accepted() : type(PM_Accepted), uid(0) {}
};
PM_Ensure(PMessage_Accepted);


struct istAlign(16) PMessage_Rejected
{
    PM_Type type;

    PMessage_Rejected() : type(PM_Rejected) {}
};
PM_Ensure(PMessage_Rejected);


struct istAlign(16) PMessage_Leave
{
    PM_Type type;

    PMessage_Leave() : type(PM_Leave) {}
};
PM_Ensure(PMessage_Leave);


struct istAlign(16) PMessage_Update
{
    PM_Type type;

    PMessage_Update() : type(PM_Update) {}
};
PM_Ensure(PMessage_Update);


struct istAlign(16) PMessage_Text
{
    PM_Type type;

    PMessage_Text() : type(PM_Text) {}
};
PM_Ensure(PMessage_Text);


struct istAlign(16) PMessage_Sync
{
    PM_Type type;
    uint32 data_size;
    void *data;

    PMessage_Sync() : type(PM_Sync), data_size(0), data(NULL) {}
};
PM_Ensure(PMessage_Sync);


#undef PM_Ensure

} // namespace atomic

#endif // atomic_Network_PMessage_h
