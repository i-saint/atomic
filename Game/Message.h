#ifndef __atomic_Message_h__
#define __atomic_Message_h__

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

enum BULLET_TYPE
{
    BULLET_OCTAHEDRON,

    BULLET_END,
};

enum FORCE_TYPE
{
    FORCE_SPHERICAL_FIELD,
    FORCE_CYLINDRICAL_FIELD,
    FORCE_CUBIC_FIELD,
    FORCE_INVERTED_SPHERICAL_FIELD,
    FORCE_INVERTED_CYLINDRICAL_FIELD,
    FORCE_INVERTED_CUBIC_FIELD,

    FORCE_SPHERICAL_GRAVITY,
    FORCE_CYLINDRICAL_GRAVITY,
    FORCE_CUBIC_GRAVITY,

    FORCE_SPHERICAL_REFLECTOR,
    FORCE_CYLINDRICAL_REFLECTOR,
    FORCE_CUBIC_REFLECTOR,

    FORCE_END,
};

enum CHARACTER_TYPE
{
    CHARACTER_TEST_PlAYER,
    CHARACTER_TEST_ENEMY,

    CHARACTER_END,
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


struct __declspec(align(16)) Message_Force
{
    uint32 force_type;
    __declspec(align(16)) char force_data[64];

    template<class ForceDataType>
    void assignData(const ForceDataType& v)
    {
        BOOST_STATIC_ASSERT(sizeof(ForceDataType)<=sizeof(force_data));
        reinterpret_cast<ForceDataType&>(*force_data) = v;
    }
};


struct __declspec(align(16)) Message_GenerateFraction
{
    enum GEN_TYPE
    {
        GEN_SPHERE,
        GEN_BOX,

        GEN_END,
    };
    uint32 gen_type;
    uint32 num;
    __declspec(align(16)) char shape_data[64];

    template<class ShapeDataType>
    void assignData(const ShapeDataType& v)
    {
        BOOST_STATIC_ASSERT(sizeof(ShapeDataType)<=sizeof(shape_data));
        reinterpret_cast<ShapeDataType&>(*shape_data) = v;
    }
};

struct __declspec(align(16)) Message_GenerateBullet
{
    BULLET_TYPE bullet_type;
    vec4 pos;
    vec4 vel;
};

struct __declspec(align(16)) Message_GenerateForce
{
    FORCE_TYPE force_type;
    vec4 pos;
};

struct __declspec(align(16)) Message_GenerateCharacter
{
    CHARACTER_TYPE character_type;
    vec4 pos;
};



enum MR_ID {
    //MR_SYSTEM,
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
#endif // __atomic_Message_h__
