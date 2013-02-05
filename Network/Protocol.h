#ifndef atomic_Network_Protocol_h
#define atomic_Network_Protocol_h
#include "externals.h"

namespace atomic {

enum PTCL_Type
{
    PTCL_Unknown,
    PTCL_Ping,
    PTCL_Pong,
    PTCL_Join,
    PTCL_Accepted,
    PTCL_Rejected,
    PTCL_Leave,
    PTCL_Update,
    PTCL_Text,
    PTCL_Sync,
};

union Protocol
{
    struct {
        PTCL_Type type;
    };
    uint32 dummy[8];

    Protocol() : type(PTCL_Unknown) {}
};

struct Protocol_Ping
{
    PTCL_Type type;

    Protocol_Ping() : type(PTCL_Ping) {}
};

struct Protocol_Pong
{
    PTCL_Type type;

    Protocol_Pong() : type(PTCL_Pong) {}
};

struct Protocol_Join
{
    PTCL_Type type;

    Protocol_Join() : type(PTCL_Join) {}
};

struct Protocol_Accepted
{
    PTCL_Type type;
    uint32 uid;

    Protocol_Accepted() : type(PTCL_Accepted), uid(0) {}
};

struct Protocol_Rejected
{
    PTCL_Type type;

    Protocol_Rejected() : type(PTCL_Rejected) {}
};

struct Protocol_Leave
{
    PTCL_Type type;

    Protocol_Leave() : type(PTCL_Leave) {}
};

struct Protocol_Update
{
    PTCL_Type type;

    Protocol_Update() : type(PTCL_Update) {}
};

struct Protocol_Text
{
    PTCL_Type type;

    Protocol_Text() : type(PTCL_Text) {}
};

struct Protocol_Sync
{
    PTCL_Type type;
    uint32 data_size;
    void *data;

    Protocol_Sync() : type(PTCL_Sync), data_size(0), data(NULL) {}
};


} // namespace atomic

#endif // atomic_Network_Protocol_h
