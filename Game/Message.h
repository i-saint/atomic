#ifndef __atomic_Message__
#define __atomic_Message__

namespace atomic
{


enum MESSAGE_TYPE
{
    MES_NONE,

    MES_KILL,
    MES_DESTROY,

    MES_COLLISION_FRACTION_FRACTION,

    MES_GENERATE_PLAYER,
    MES_GENERATE_GROUND,
    MES_GENERATE_ENEMY,
    MES_GENERATE_FRACTION,
    MES_GENERATE_FORCE,
    MES_GENERATE_VFX,
};


struct Message
{
    int32 type;
};

struct Message_Kill
{
    int32 type;
    id_t receiver;
};

struct Message_Destroy
{
    int32 type;
    id_t receiver;
};


struct Message_GenerateFraction
{
    enum GEN_TYPE
    {
        GEN_POINT,
        GEN_SPHERE,
        GEN_BOX,
    };
    int32 type;
    int32 gen_type;
    uint32 num;
    __declspec(align(16)) char shape_data[sizeof(ist::OBB)];
};


void SendKillMessage(id_t receiver);
void SendDestroyMessage(id_t receiver);

} // namespace scntilla
#endif // __atomic_Message__
