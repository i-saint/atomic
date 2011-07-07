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


enum MR_ID {
    MR_SYSTEM,
    MR_FRACTION,
    //MR_BULLET,
    //MR_CHARACTER,
    //MR_FORCE,

    MR_END,
};

class MessageRouter : boost::noncopyable
{
public:
    enum STATUS
    {
        ST_RECEIVING,   // メッセージ蓄積中。オーナー以外触れちゃダメ
        ST_ROUTING,     // メッセージ配送中。誰でも触れられる
    };

    class MessageBlock
    {
    private:
        stl::vector<Message_GenerateFraction> m_gen_fraction;

    public:
        template<class MessageType>
        void push(MR_ID id, uint32 block, const Message_GenerateFraction& mes);

        template<class MessageType>
        void get(MR_ID id, uint32 block, const Message_GenerateFraction& mes);
    };

private:
    stl::vector<MessageBlock*> m_blocks[MR_END];
    uint32 m_user_flag; // 各ビットが MR_ID のオーナーに対応 
    STATUS m_status;

    static MessageRouter *s_instance[MR_END];

    MessageRouter();

public:
    ~MessageRouter();
    static void initializeInstance();
    static void finalizeInstance();
    static MessageRouter* getInstance(MR_ID id);

    STATUS getStatus() const { return m_status; }
    uint32 getMessageBlockNum(MR_ID id) const;
    void resizeMessageBlock(MR_ID id, uint32 num);
    MessageBlock* getMessageBlock(MR_ID id, uint32 i=0);

    void beginReceive();
    void endReceive();
    void beginRoute();
    void endRoute(MR_ID id);
    static void endRouteAll(MR_ID id);
};

} // namespace atomic
#endif // __atomic_Message__
