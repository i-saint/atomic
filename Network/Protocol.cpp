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



PMessage_Ping PMessage_Ping::create()
{
    PMessage_Ping t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_Ping;
    return t;
}

PMessage_Pong PMessage_Pong::create()
{
    PMessage_Pong t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_Pong;
    return t;
}

PMessage_Accepted PMessage_Accepted::create(PlayerID pid)
{
    PMessage_Accepted t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_Accepted;
    t.player_id = pid;
    return t;
}

PMessage_Rejected PMessage_Rejected::create()
{
    PMessage_Rejected t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_Rejected;
    return t;
}

PMessage_Join PMessage_Join::create( PlayerID pid, const PlayerName &name )
{
    PMessage_Join t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_Join;
    t.player_id = pid;
    t.frame = 0;
    memcpy(t.name, name, sizeof(PlayerName));
    return t;
}


PMessage_Leave PMessage_Leave::create( PlayerID pid )
{
    PMessage_Leave t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_Leave;
    t.player_id = pid;
    return t;
}

PMessage_Update PMessage_Update::create( PlayerID pid, uint32 frame, const RepInput &inp )
{
    PMessage_Update t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_Update;
    t.player_id = pid;
    t.frame = frame;
    t.input = inp;
    return t;
}

PMessage_LEC PMessage_LEC::create( const LevelEditorCommand &lec )
{
    PMessage_LEC t;
    istMemset(&t, 0, sizeof(t));
    t.type = PM_LevelEditorCommand;
    t.lec = lec;
    return t;
}




bool SendPMessages( Poco::Net::SocketStream *stream, PMessageBuffer &buf, PMessageCont &messages )
{
    buf.clear();
    buf.resize( sizeof(PMesBufferHeader) );
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
    stream->write(&buf[0], buf.size());
    stream->flush();
    return true;
}

bool RecvPMessages( Poco::Net::SocketStream *stream, PMessageBuffer &buf, PMessageCont &messages )
{
    messages.clear();
    buf.resize(sizeof(PMesBufferHeader));
    stream->read(&buf[0], buf.size());
    if(stream->eof()) { return false; }

    PMesBufferHeader header = *(PMesBufferHeader*)&buf[0];
    istAssert(strcmp(header.magic, "atomic")==0);
    if(header.length_in_byte==0) { return true; }
    buf.resize(header.length_in_byte);

    stream->read(&buf[0], buf.size());
    if(stream->eof()) { return false; }
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

void PMessenger::pushMessage( const PMessage &p )
{
    ist::Mutex::ScopedLock lock(m_mutex_send);
    m_message_send.push_back(p);
}

void PMessenger::pushMessage( const PMessageCont &p )
{
    if(p.empty()) { return; }
    pushMessage(&p[0], p.size());
}

void PMessenger::pushMessage( const PMessage *p, size_t num )
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

void PMessenger::handleReceivedMessageCont( const MessageContHandler &h )
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_recv);
        m_message_consuming = m_message_recv;
        m_message_recv.clear();
    }
    h(m_message_consuming);
    m_message_consuming.clear();
}

bool PMessenger::sendMessage(Poco::Net::SocketStream *stream)
{
    for(int32 i=0; i<10; ++i) {
        if(m_message_send.empty()) {
            ist::Thread::milliSleep(2);
        }
    }
    {
        ist::Mutex::ScopedLock lock(m_mutex_send);
        m_message_sending = m_message_send;
        m_message_send.clear();
    }
    bool ret = SendPMessages(stream, m_message_buffer, m_message_sending);
    DestructMessages(m_message_sending);
    return ret;
}

bool PMessenger::recvMessage(Poco::Net::SocketStream *stream)
{
    bool ret = RecvPMessages(stream, m_message_buffer, m_message_receiving);
    if(ret && !m_message_receiving.empty()) {
        processReceivingMessage(m_message_receiving);
    }
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
