#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"


namespace atomic {

MessageRouter* MessageRouter::s_instance[MR_END];

MessageRouter::MessageRouter()
: m_user_flag(0)
, m_status(ST_RECEIVING)
{
    for(uint32 i=0; i<_countof(m_blocks); ++i) {
        m_blocks[i].push_back(AT_NEW(MessageBlock)());
    }
}

MessageRouter::~MessageRouter()
{
    for(uint32 i=0; i<_countof(m_blocks); ++i) {
        for(uint32 bi=0; bi<m_blocks[i].size(); ++bi) {
            AT_DELETE(m_blocks[i][bi]);
        }
    }
}

void MessageRouter::initializeInstance()
{
    for(uint32 i=0; i<_countof(s_instance); ++i) {
        s_instance[i] = AT_NEW(MessageRouter)();
    }
}

void MessageRouter::finalizeInstance()
{
    for(uint32 i=0; i<_countof(s_instance); ++i) {
        AT_DELETE(s_instance[i]);
    }
}

MessageRouter* MessageRouter::getInstance( MR_ID id )
{
    return s_instance[id];
}

uint32 MessageRouter::getMessageBlockNum(MR_ID id) const
{
    return m_blocks[id].size();
}

void MessageRouter::resizeMessageBlock(MR_ID id, uint32 num)
{
    while(m_blocks[id].size() < num) {
        m_blocks[id].push_back(AT_NEW(MessageBlock)());
    }
}

MessageRouter::MessageBlock* MessageRouter::getMessageBlock(MR_ID id, uint32 i)
{
    return m_blocks[id][i];
}

void MessageRouter::beginReceive()
{
    while(m_status==ST_ROUTING) {
        // wait
    }
}

void MessageRouter::endReceive()
{
    // currently do nothing
}

void MessageRouter::beginRoute()
{
    m_user_flag = (1<<MR_END)-1;
}

void MessageRouter::endRoute( MR_ID id )
{
    m_user_flag = m_user_flag & ~(1<<id);
    if(m_user_flag==0) {
        m_status = ST_RECEIVING;
    }
}

void MessageRouter::endRouteAll( MR_ID id )
{
    for(uint32 i=0; i<MR_END; ++i) {
        getInstance((MR_ID)i)->endRoute(id);
    }
}

} // namespace atomic