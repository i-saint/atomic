#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"


namespace atomic {


template<> stl::vector<Message_Kill>*               MessageRouter::MessageBlock::getContainer<Message_Kill>()               { return &m_mes_kill; }
template<> stl::vector<Message_Destroy>*            MessageRouter::MessageBlock::getContainer<Message_Destroy>()            { return &m_mes_destroy; }
template<> stl::vector<Message_Force>*              MessageRouter::MessageBlock::getContainer<Message_Force>()              { return &m_mes_force; }
template<> stl::vector<Message_GenerateFraction>*   MessageRouter::MessageBlock::getContainer<Message_GenerateFraction>()   { return &m_mes_genfraction; }
template<> stl::vector<Message_GenerateBullet>*     MessageRouter::MessageBlock::getContainer<Message_GenerateBullet>()     { return &m_mes_genbullet; }
template<> stl::vector<Message_GenerateCharacter>*  MessageRouter::MessageBlock::getContainer<Message_GenerateCharacter>()  { return &m_mes_gencharacter; }
template<> stl::vector<Message_GenerateForce>*      MessageRouter::MessageBlock::getContainer<Message_GenerateForce>()      { return &m_mes_genforce; }

template<> const stl::vector<Message_Kill>*                 MessageRouter::MessageBlock::getContainer<Message_Kill>() const             { return &m_mes_kill; }
template<> const stl::vector<Message_Destroy>*              MessageRouter::MessageBlock::getContainer<Message_Destroy>() const          { return &m_mes_destroy; }
template<> const stl::vector<Message_Force>*                MessageRouter::MessageBlock::getContainer<Message_Force>() const            { return &m_mes_force; }
template<> const stl::vector<Message_GenerateFraction>*     MessageRouter::MessageBlock::getContainer<Message_GenerateFraction>() const { return &m_mes_genfraction; }
template<> const stl::vector<Message_GenerateBullet>*       MessageRouter::MessageBlock::getContainer<Message_GenerateBullet>() const   { return &m_mes_genbullet; }
template<> const stl::vector<Message_GenerateCharacter>*    MessageRouter::MessageBlock::getContainer<Message_GenerateCharacter>() const{ return &m_mes_gencharacter; }
template<> const stl::vector<Message_GenerateForce>*        MessageRouter::MessageBlock::getContainer<Message_GenerateForce>() const    { return &m_mes_genforce; }

void MessageRouter::MessageBlock::clear()
{
    m_mes_kill.clear();
    m_mes_destroy.clear();
    m_mes_force.clear();
    m_mes_genfraction.clear();
    m_mes_genbullet.clear();
    m_mes_gencharacter.clear();
    m_mes_genforce.clear();
}




MessageRouter* MessageRouter::s_instance[MR_END];

void MessageRouter::initializeInstance()
{
    for(uint32 i=0; i<_countof(s_instance); ++i) {
        s_instance[i] = IST_NEW(MessageRouter)((MR_ID)i);
    }
}

void MessageRouter::finalizeInstance()
{
    for(uint32 i=0; i<_countof(s_instance); ++i) {
        IST_DELETE(s_instance[i]);
    }
}

MessageRouter* MessageRouter::getInstance(MR_ID id)
{
    return s_instance[id];
}


MessageRouter::MessageRouter(MR_ID owner)
: m_front_block(NULL)
, m_back_block(NULL)
, m_user_flag((1<<MR_END)-1)
, m_status(ST_ROUTING)
, m_owner(owner)
{
    for(uint32 i=0; i<_countof(m_blocks); ++i) {
        m_blocks[i].push_back(IST_NEW(MessageBlock)());
    }
    m_front_block = &m_blocks[0];
    m_back_block = &m_blocks[1];
}

MessageRouter::~MessageRouter()
{
    for(uint32 i=0; i<_countof(m_blocks); ++i) {
        for(uint32 bi=0; bi<m_blocks[i].size(); ++bi) {
            IST_DELETE(m_blocks[i][bi]);
        }
    }
}

uint32 MessageRouter::getMessageBlockNum() const
{
    return m_front_block->size();
}

void MessageRouter::resizeMessageBlock(uint32 num)
{
    for(uint32 i=0; i<_countof(m_blocks); ++i) {
        while(m_blocks[i].size() < num) {
            m_blocks[i].push_back(IST_NEW(MessageBlock)());
        }
    }
}

const MessageRouter::MessageBlock* MessageRouter::getMessageBlock(uint32 i) const
{
    while(m_status != ST_ROUTING) {
        TaskScheduler::wait();
    }
    return (*m_front_block)[i];
}

MessageRouter::MessageBlock* MessageRouter::getMessageBlockForWrite(uint32 i)
{
    return (*m_back_block)[i];
}


void MessageRouter::route()
{
    while(m_status!=ST_ROUTE_COMPLETE) {
        TaskScheduler::wait();
    }

    stl::swap<MessageBlockCont*>(m_front_block, m_back_block);
    for(uint32 i=0; i<m_back_block->size(); ++i) {
        (*m_back_block)[i]->clear();
    }

    m_lock_status.lock();
    m_user_flag = (1<<MR_END)-1;
    m_status = ST_ROUTING;
    m_lock_status.unlock();
}

void MessageRouter::unuse( MR_ID id )
{
    while(m_status != ST_ROUTING) {
        TaskScheduler::wait();
    }

    m_lock_status.lock();
    m_user_flag = m_user_flag & ~(1<<id);
    if(m_user_flag==0) {
        m_status = ST_ROUTE_COMPLETE;
    }
    m_lock_status.unlock();
}

void MessageRouter::unuseAll()
{
    for(uint32 i=0; i<MR_END; ++i) {
        getInstance((MR_ID)i)->unuse(m_owner);
    }
}



template<class T>
atomic::MessageIterator<T>::MessageIterator()
: m_router_index(0)
, m_block_index(0)
, m_cont_index(0)
, m_block_size(0)
, m_cont_size(0)
{
    m_router = atomicGetMessageRouter((MR_ID)m_router_index);
    m_block = m_router->getMessageBlock(m_block_index);
    m_block_size = m_router->getMessageBlockNum();
    m_cont = m_block->getContainer<MessageType>();
    m_cont_size = m_cont->size();
}

template<class T>
bool atomic::MessageIterator<T>::hasNext()
{
    for(; m_router_index<MR_END; ++m_router_index) {
        for(; m_block_index<m_block_size; ++m_block_index) {
            if(m_cont_index < m_cont_size) {
                return true;
            }
            if(m_block_index+1==m_block_size) {
                break;
            }
            m_block = m_router->getMessageBlock(m_block_index+1);
            m_cont = m_block->getContainer<MessageType>();
            m_cont_index = 0;
            m_cont_size = m_cont->size();
        }
        if(m_router_index+1==MR_END) {
            break;
        }
        m_router = atomicGetMessageRouter((MR_ID)(m_router_index+1));
        m_block_index = 0;
        m_block_size = m_router->getMessageBlockNum();
        m_block = m_router->getMessageBlock(m_block_index);
        m_cont = m_block->getContainer<MessageType>();
        m_cont_index = 0;
        m_cont_size = m_cont->size();
    }

    return false;
}

template<class T>
const T& atomic::MessageIterator<T>::iterate()
{
    return (*m_cont)[m_cont_index++];
}

template MessageIterator<Message_Kill>;
template MessageIterator<Message_Destroy>;
template MessageIterator<Message_Force>;
template MessageIterator<Message_GenerateFraction>;
template MessageIterator<Message_GenerateBullet>;
template MessageIterator<Message_GenerateCharacter>;
template MessageIterator<Message_GenerateForce>;


} // namespace atomic