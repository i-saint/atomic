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

} // namespace atomic
