#include "stdafx.h"
#include "Protocol.h"

namespace atomic {


void PMessage::destroy()
{
    switch(type) {
    case PM_Sync:
        break;

    default:
        break;
    }
}



bool SendPMessages( Poco::Net::StreamSocket *stream, PMessageBuffer &buf, PMessageCont &messages )
{
    buf.clear();
    buf.resize(sizeof(PMesBufferHeader));
    {
        PMesBufferHeader &header = *(PMesBufferHeader*)&buf[0];
        memcpy(header.magic, PM_message_header, _countof(header.magic));
        header.version = atomic_version;
        header.num_message = messages.size();
        // 途中の insert で↑の header は無効になる可能性があるため、ここで scope 切る
    }

    for(size_t i=0; i<messages.size(); ++i) {
        const PMessage &mes = messages[i];
        const char *data = (char*)&mes;
        buf.insert(buf.end(), data, data+sizeof(PMessage));
        // todo: 可変長メッセージ対策
    }

    {
        PMesBufferHeader &header = *(PMesBufferHeader*)&buf[0];
        header.length_in_byte = buf.size()-sizeof(PMesBufferHeader);
    }
    stream->sendBytes(&buf[0], buf.size());
    return true;
}

bool RecvPMessages( Poco::Net::StreamSocket *stream, PMessageBuffer &buf, PMessageCont &messages )
{
    messages.clear();
    buf.resize(sizeof(PMesBufferHeader));
    if(stream->receiveBytes(&buf[0], buf.size())==0) { return false; }

    PMesBufferHeader header = *(PMesBufferHeader*)&buf[0];
    buf.resize(header.length_in_byte);
    if(stream->receiveBytes(&buf[0], buf.size())==0) { return false; }
    const PMessage *pmes = (PMessage*)(&buf[0]);
    for(size_t i=0; i<header.num_message; ++i) {
        messages.push_back(pmes[i]);
    }
    return true;
}

void DestructMessages( PMessageCont &messages )
{
    for(size_t i=0; i<messages.size(); ++i) {
        messages[i].destroy();
    }
    messages.clear();
}





PMessenger::~PMessenger()
{
    clearAllMessage();
}

void PMessenger::queueMessage( const PMessage &p )
{
    ist::Mutex::ScopedLock lock(m_mutex_send);
    m_message_send.push_back(p);
}

void PMessenger::queueMessage( const PMessage *p, size_t num )
{
    ist::Mutex::ScopedLock lock(m_mutex_send);
    m_message_send.insert(m_message_send.end(), p, p+num);
}

void PMessenger::handleReceivedMessage( const MessageHandler &h )
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_recv);
        m_message_consuming = m_message_recv;
        m_message_recv.clear();
    }
    for(size_t i=0; i<m_message_consuming.size(); ++i) {
        h(m_message_consuming[i]);
    }
    DestructMessages(m_message_consuming);
}

bool PMessenger::sendMessage(Poco::Net::StreamSocket *stream)
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_send);
        m_message_sending = m_message_send;
        m_message_send.clear();
    }
    bool ret = SendPMessages(stream, m_message_buffer, m_message_sending);
    DestructMessages(m_message_sending);
    return ret;
}

bool PMessenger::recvMessage(Poco::Net::StreamSocket *stream)
{
    bool ret = RecvPMessages(stream, m_message_buffer, m_message_receiving);
    {
        ist::Mutex::ScopedLock lock(m_mutex_recv);
        m_message_recv.insert(m_message_recv.end(), m_message_receiving.begin(), m_message_receiving.end());
    }
    m_message_receiving.clear();
    return ret;
}

void PMessenger::clearAllMessage()
{
    ist::Mutex::ScopedLock slock(m_mutex_send);
    ist::Mutex::ScopedLock rlock(m_mutex_recv);
    DestructMessages(m_message_send);
    DestructMessages(m_message_recv);
}

} // namespace atomic
