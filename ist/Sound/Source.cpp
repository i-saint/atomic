#include "stdafx.h"
#include "../Sound.h"

namespace ist {
namespace sound {

int Source::getI(ALenum param) const
{
    int v = 0;
    alGetSourcei(m_handle, param, &v);
    return v;
}

float Source::getF(ALenum param) const
{
    float r;
    alGetSourcef(m_handle, param, &r);
    return r;
}

vec3 Source::get3F(ALenum param) const
{
    vec3 r;
    alGetSourcefv(m_handle, param, (ALfloat*)&r);
    return r;
}

void Source::setI(ALenum param, int v)
{
    alSourcei(m_handle, param, v);
}

void Source::setF(ALenum param, float v)
{
    alSourcef(m_handle, param, v);
}

void Source::set3F(ALenum param, const vec3& v)
{
    alSourcefv(m_handle, param, (ALfloat*)&v);
}


Source::Source() : m_handle(0)
{
    alGenSources(1, &m_handle);
}

Source::~Source()
{
    alDeleteSources(1, &m_handle);
}


bool Source::isInitial() const { return getI(AL_SOURCE_STATE)==AL_INITIAL; }
bool Source::isPlaying() const { return getI(AL_SOURCE_STATE)==AL_PLAYING; }
bool Source::isPaused() const  { return getI(AL_SOURCE_STATE)==AL_PAUSED; }
bool Source::isStopped() const { return getI(AL_SOURCE_STATE)==AL_STOPPED; }
int Source::getProcessed() const { return getI(AL_BUFFERS_PROCESSED); }

void Source::play()   { alSourcePlay(m_handle); }
void Source::pause()  { alSourcePause(m_handle); }
void Source::stop()   { alSourceStop(m_handle); }
void Source::rewind() { alSourceRewind(m_handle); }


bool Source::unqueue()
{
    ALuint buf = 0;
    alSourceUnqueueBuffers(m_handle, 1, &buf);
    return buf!=0;
}

void Source::queue(Buffer *buf)
{
    ALuint h = buf->getHandle();
    alSourceQueueBuffers(m_handle, 1, &h);
}

void Source::clearQueue()
{
    setI(AL_BUFFER, AL_NONE);
}


} // namespace sound
} // namespace ist
