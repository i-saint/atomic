#include "istPCH.h"
#include "../Sound.h"

namespace ist {
namespace isd {

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


Source::Source()
{
    alGenSources(1, &m_handle);
}

Source::~Source()
{
    alDeleteSources(1, &m_handle);
}

bool Source::isLooping() const              { return getI(AL_LOOPING)==AL_TRUE; }
float Source::getGain() const               { return getF(AL_GAIN); }
float Source::getRefferenceDistance() const { return getF(AL_REFERENCE_DISTANCE); }
float Source::getRolloffFactor() const      { return getF(AL_ROLLOFF_FACTOR); }
float Source::getMaxDistance() const        { return getF(AL_MAX_DISTANCE); }
float Source::getPitch() const              { return getF(AL_PITCH); }
vec3 Source::getPosition() const            { return get3F(AL_POSITION); }
vec3 Source::getVelocity() const            { return get3F(AL_VELOCITY); }
int Source::getNumQueuedBuffers() const     { return getI(AL_BUFFERS_QUEUED); }
int Source::getNumProcessedBuffers() const  { return getI(AL_BUFFERS_PROCESSED); }

void Source::setLooping(bool v)             { setI(AL_LOOPING, v); }
void Source::setGain(float v)               { setF(AL_GAIN, v); }
void Source::setRefferenceDistance(float v) { setF(AL_REFERENCE_DISTANCE, v); }
void Source::setRolloffFactor(float v)      { setF(AL_ROLLOFF_FACTOR, v); }
void Source::setMaxDistance(float v)        { setF(AL_MAX_DISTANCE, v); }
void Source::setPitch(float v)              { setF(AL_PITCH, v); }
void Source::setPosition(const vec3& v)     { set3F(AL_POSITION, v); }
void Source::setVelocity(const vec3& v)     { set3F(AL_VELOCITY, v); }

bool Source::isInitial() const  { return getI(AL_SOURCE_STATE)==AL_INITIAL; }
bool Source::isPlaying() const  { return getI(AL_SOURCE_STATE)==AL_PLAYING; }
bool Source::isPaused() const   { return getI(AL_SOURCE_STATE)==AL_PAUSED; }
bool Source::isStopped() const  { return getI(AL_SOURCE_STATE)==AL_STOPPED; }

void Source::play()     { alSourcePlay(m_handle); }
void Source::pause()    { alSourcePause(m_handle); }
void Source::stop()     { alSourceStop(m_handle); }
void Source::rewind()   { alSourceRewind(m_handle); }


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
    rewind();
    setI(AL_BUFFER, AL_NONE);
}


} // namespace isd
} // namespace ist
