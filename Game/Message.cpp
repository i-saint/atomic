#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/Entity.h"
#include "Message.h"


namespace atomic {


MessageRouter* MessageRouter::s_instance;

void MessageRouter::initializeInstance()
{
    s_instance = istNew(MessageRouter)();
}

void MessageRouter::finalizeInstance()
{
    istSafeDelete(s_instance);
}

MessageRouter* MessageRouter::getInstance()
{
    return s_instance;
}


MessageRouter::MessageRouter()
{
}

MessageRouter::~MessageRouter()
{
}

uint32 MessageRouter::getMessageBlockNum() const
{
    return m_blocks.size();
}

void MessageRouter::resizeMessageBlock(uint32 num)
{
    while(m_blocks.size() < num) {
        m_blocks.push_back(istNew(MessageBlock)());
    }
}

MessageRouter::MessageBlock* MessageRouter::getMessageBlock(uint32 i)
{
    return m_blocks[i];
}

void MessageRouter::clear()
{
    for(uint32 i=0; i<m_blocks.size(); ++i) {
        m_blocks[i]->clear();
    }
}



MessageIterator::MessageIterator()
: m_num_blocks(0)
, m_num_messages(0)
, m_block_index(0)
, m_message_index(0)
, m_current_block(NULL)
{
    MessageRouter *router = atomicGetMessageRouter();
    m_num_blocks = router->getMessageBlockNum();
    m_current_block = router->getMessageBlock(0);
    m_num_messages = m_current_block->size();
}

bool atomic::MessageIterator::hasNext()
{
    MessageRouter *router = atomicGetMessageRouter();
    for(; m_block_index<m_num_blocks; ++m_block_index) {
        if(m_message_index < m_num_messages) {
            return true;
        }
        if(m_block_index+1==m_num_blocks) {
            break;
        }
        m_current_block = router->getMessageBlock(m_block_index+1);
        m_num_messages = m_current_block->size();
        m_message_index = 0;
    }
    return false;
}

const CallInfo& atomic::MessageIterator::iterate()
{
    return (*m_current_block)[m_message_index++];
}


} // namespace atomic