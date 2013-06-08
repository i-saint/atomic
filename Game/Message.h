#ifndef atm_Message_h
#define atm_Message_h

namespace atm {


struct CallInfo
{
    EntityHandle from;
    EntityHandle to;
    uint32 call_id;
    uint32 pad;
    variant argument;
};

struct DamageMessage
{
    EntityHandle from;
    EntityHandle to;
    float32 damage;
    int32 attribute;
};



class MessageRouter
{
istNonCopyable(MessageRouter);
public:
    typedef ist::vector<CallInfo>       MessageBlock;
    typedef ist::vector<MessageBlock*>  MessageBlockCont;

private:
    static MessageRouter *s_instance;
    MessageBlockCont m_blocks;
    SpinMutex m_lock_status;

    MessageRouter();

public:
    ~MessageRouter();
    static void initializeInstance();
    static void finalizeInstance();
    static MessageRouter* getInstance();

    uint32 getMessageBlockNum() const;
    void resizeMessageBlock(uint32 num);
    MessageBlock* getMessageBlock(uint32 i);

    void clear();
};


#define atmGetMessageRouter()            MessageRouter::getInstance()
#define atmPushCallMessage(block, mes)   atmGetMessageRouter()->getMessageBlock(block)->addTask(mes)


class MessageIterator
{
private:
    typedef MessageRouter::MessageBlock MessageBlock;
    uint32 m_num_blocks;
    uint32 m_num_messages;
    uint32 m_block_index;
    uint32 m_message_index;
    MessageBlock *m_current_block;

public:
    MessageIterator();
    bool hasNext();
    const CallInfo& iterate();
};

} // namespace atm
#endif // atm_Message_h
