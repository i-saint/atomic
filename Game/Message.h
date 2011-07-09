#ifndef __atomic_Message__
#define __atomic_Message__

namespace atomic {


enum MESSAGE_TYPE
{
    MES_KILL,
    MES_DESTROY,

    MES_DAMAGE,
    MES_FORCE,

    MES_COLLISION_FRACTION_FRACTION,

    MES_GENERATE_CHARACTER,
    MES_GENERATE_BULLET,
    MES_GENERATE_FRACTION,
    MES_GENERATE_FORCE,
    MES_GENERATE_VFX,
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


struct Message_Force
{
    uint32 force_type;
};


struct Message_GenerateFraction
{
    enum GEN_TYPE
    {
        GEN_SPHERE,
        GEN_BOX,

        GEN_END,
    };
    uint32 gen_type;
    uint32 num;
    __declspec(align(16)) char shape_data[sizeof(ist::OBB)];
};

struct Message_GenerateBullet
{
    uint32 force_type;
};

struct Message_GenerateForce
{
    uint32 force_type;
};

struct Message_GenerateCharacter
{
    uint32 force_type;
};



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
        ST_ROUTING,     // メッセージ配送中。誰でも触れられる
        ST_ROUTE_COMPLETE,  // メッセージ配送完了
    };

    class MessageBlock
    {
    private:
        stl::vector<Message_Kill>               m_mes_kill;
        stl::vector<Message_Destroy>            m_mes_destroy;
        stl::vector<Message_Force>              m_mes_force;
        stl::vector<Message_GenerateFraction>   m_mes_genfraction;
        stl::vector<Message_GenerateBullet>     m_mes_genbullet;
        stl::vector<Message_GenerateCharacter>  m_mes_gencharacter;
        stl::vector<Message_GenerateForce>      m_mes_genforce;

    public:
        template<class MessageType> stl::vector<MessageType>* getContainer();
        template<class MessageType> const stl::vector<MessageType>* getContainer() const;

        void clear();
    };

private:
    typedef stl::vector<MessageBlock*> MessageBlockCont;
    MessageBlockCont m_blocks[2];
    MessageBlockCont *m_front_block, *m_back_block;

    uint32 m_user_flag; // 各ビットが MR_ID のオーナーに対応 
    STATUS m_status;
    SpinLock m_lock_status;
    MR_ID m_owner;

    static MessageRouter *s_instance[MR_END];

    MessageRouter(MR_ID owner);

public:
    ~MessageRouter();
    static void initializeInstance();
    static void finalizeInstance();
    static MessageRouter* getInstance(MR_ID id);

    STATUS getStatus() const { return m_status; }
    uint32 getMessageBlockNum() const;
    void resizeMessageBlock(uint32 num);
    const MessageBlock* getMessageBlock(uint32 i) const;
    MessageBlock* getMessageBlockForWrite(uint32 i);

    void route();
    void unuse(MR_ID id);
    void unuseAll();
};


#define atomicGetMessageRouter(id) MessageRouter::getInstance(id)

template<class T>
inline void atomicPushMessage(MR_ID id, uint32 block, const T& mes)
{
    atomicGetMessageRouter(id)->getMessageBlockForWrite(block)->getContainer<T>()->push_back(mes);
}


template<class T>
class MessageIterator
{
public:
    typedef T MessageType;
    typedef stl::vector<MessageType> MessageContainer;
    typedef MessageRouter::MessageBlock MessageBlock;

private:
    int32 m_router_index;
    int32 m_block_index;
    int32 m_cont_index;
    int32 m_block_size;
    int32 m_cont_size;
    const MessageRouter *m_router;
    const MessageBlock *m_block;
    const MessageContainer *m_cont;

public:
    MessageIterator();
    bool hasNext();
    const MessageType& iterate();
};

} // namespace atomic
#endif // __atomic_Message__
