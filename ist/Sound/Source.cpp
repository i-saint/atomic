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

ALuint Source::getHandle() const { return m_handle; }


float Source::getGain() const           { return getF(AL_GAIN); }
vec3 Source::getPosition() const        { return get3F(AL_POSITION); }
vec3 Source::getVelocity() const        { return get3F(AL_VELOCITY); }

void Source::setGain(float v)           { setF(AL_GAIN, v); }
void Source::setPosition(const vec3& v) { set3F(AL_POSITION, v); }
void Source::setVelocity(const vec3& v) { set3F(AL_VELOCITY, v); }


bool Source::isInitial() const { return getI(AL_SOURCE_STATE)==AL_INITIAL; }
bool Source::isPlaying() const { return getI(AL_SOURCE_STATE)==AL_PLAYING; }
bool Source::isPaused() const  { return getI(AL_SOURCE_STATE)==AL_PAUSED; }
bool Source::isStopped() const { return getI(AL_SOURCE_STATE)==AL_STOPPED; }
int Source::getProcessed() const { return getI(AL_BUFFERS_PROCESSED); }

void Source::play()   { alSourcePlay(m_handle); }
void Source::pause()  { alSourcePause(m_handle); }
void Source::stop()   { alSourceStop(m_handle); }
void Source::rewind() { alSourceRewind(m_handle); }


BufferPtr Source::unqueue()
{
    if(m_queue.empty()) {
        return BufferPtr();
    }
    ALuint buf = 0;
    alSourceUnqueueBuffers(m_handle, 1, &buf);
    BufferPtr front = m_queue.front();
    m_queue.pop_front();
    return front;
}

void Source::queue(BufferPtr buf)
{
    m_queue.push_back(buf);
    ALuint h = buf->getHandle();
    alSourceQueueBuffers(m_handle, 1, &h);
}

void Source::update() {}


} // namespace sound
} // namespace ist
