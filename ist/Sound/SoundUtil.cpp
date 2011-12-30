#include "stdafx.h"
#include <string>
#include "../Sound.h"
#include "../Base.h"

namespace ist {
namespace sound {

bool CreateBufferFromWaveFile(const char* filepath, Buffer *buf)
{
    Stream *s = CreateStreamFromWaveFile(filepath);
    if(s) {
        std::vector<char>& tmp = s->readByte(s->size());
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
        std::vector<char>& tmp = s->readByte(s->size());
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

} // namespace sound
} // namespace ist
