#ifndef __atomic_Message_h__
#define __atomic_Message_h__

namespace atomic {


struct __declspec(align(16)) CallMessage
{
    EntityHandle from;
    EntityHandle to;
    uint32 call_id;
    uint32 pad;
    variant argument;
};



class MessageRouter : boost::noncopyable
{
public:
    typedef stl::vector<CallMessage>    MessageBlock;
    typedef stl::vector<MessageBlock*>  MessageBlockCont;

private:
    static MessageRouter *s_instance;
    MessageBlockCont m_blocks;
    SpinLock m_lock_status;

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


#define atomicGetMessageRouter()            MessageRouter::getInstance()
#define atomicPushCallMessage(block, mes)   atomicGetMessageRouter()->getMessageBlock(block)->push_back(mes)


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
    const CallMessage& iterate();
};

} // namespace atomic
#endif // __atomic_Message_h__
