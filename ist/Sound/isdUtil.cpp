#include "istPCH.h"
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
#ifdef ist_with_oggvorbis
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
#ifdef ist_with_oggvorbis
    OggVorbisFileStream *stream = new OggVorbisFileStream();
    if(stream->openStream(filepath)) {
        return stream;
    }
#endif
    return NULL;
}



} // namespace isd
} // namespace ist
