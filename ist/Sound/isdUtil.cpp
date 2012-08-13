#include "stdafx.h"
#include <string>
#include "../Sound.h"
#include "../Base.h"

namespace ist {
namespace isd {

bool CreateBufferFromWaveFile(const char* filepath, Buffer *buf)
{
    Stream *s = CreateStreamFromWaveFile(filepath);
    if(s) {
        stl::vector<char>& tmp = s->readByte(s->size());
        buf->copy(&tmp[0], tmp.size(), s->getALFormat(), s->getSampleRate());
        delete s;
        return true;
    }
    return false;
}

bool CreateBufferFromOggFile(const char* filepath, Buffer *buf)
{
#ifdef __ist_with_oggvorbis__
    Stream *s = CreateStreamFromOggFile(filepath);
    if(s) {
        stl::vector<char>& tmp = s->readByte(s->size());
        buf->copy(&tmp[0], tmp.size(), s->getALFormat(), s->getSampleRate());
        delete s;
        return true;
    }
#endif
    return false;
}

Stream* CreateStreamFromWaveFile(const char* filepath)
{
    WaveStream *stream = new WaveStream();
    if(stream->openStream(filepath)) {
        return stream;
    }
    return NULL;
}

Stream* CreateStreamFromOggFile(const char* filepath)
{
#ifdef __ist_with_oggvorbis__
    OggVorbisFileStream *stream = new OggVorbisFileStream();
    if(stream->openStream(filepath)) {
        return stream;
    }
#endif
    return NULL;
}


ReferenceCounter::ReferenceCounter() : m_reference_count(0) {}
uint32 ReferenceCounter::getRef() const { return m_reference_count; }
uint32 ReferenceCounter::addRef() { return ++m_reference_count; }
uint32 ReferenceCounter::release()
{
    if(--m_reference_count==0) {
        istDelete(this);
    }
    return m_reference_count;
}

} // namespace isd
} // namespace ist
