#include "stdafx.h"
#include <string>
#include "../Sound.h"
#include "../Base.h"

namespace ist {
namespace sound {

BufferPtr CreateBufferFromWaveFile(const char* filepath)
{
    StreamPtr s = CreateStreamFromWaveFile(filepath);
    if(s) {
        std::vector<char>& tmp = s->readByte(s->size());
        BufferPtr ret(new Buffer());
        ret->copy(&tmp[0], tmp.size(), s->getALFormat(), s->getSampleRate());
        return ret;
    }
    return BufferPtr();
}

BufferPtr CreateBufferFromOggFile(const char* filepath)
{
#ifdef IST_SOUND_ENABLE_OGGVORBIS
    StreamPtr s = CreateStreamFromOggFile(filepath);
    if(s) {
        std::vector<char>& tmp = s->readByte(s->size());
        BufferPtr ret(new Buffer());
        ret->copy(&tmp[0], tmp.size(), s->getALFormat(), s->getSampleRate());
        return ret;
    }
#endif
    return BufferPtr();
}

StreamPtr CreateStreamFromWaveFile(const char* filepath)
{
    WaveStream *stream = new WaveStream();
    StreamPtr ret(stream);
    if(stream->openStream(filepath)) {
        return ret;
    }
    return StreamPtr();
}

StreamPtr CreateStreamFromOggFile(const char* filepath)
{
#ifdef IST_SOUND_ENABLE_OGGVORBIS
    OggVorbisFileStream *stream = new OggVorbisFileStream();
    StreamPtr ret(stream);
    if(stream->openStream(filepath)) {
        return ret;
    }
#endif
    return StreamPtr();
}

} // namespace sound
} // namespace ist
