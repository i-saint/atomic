#include "istPCH.h"
#include "../Base.h"
#include "../Sound.h"
#include "isdDevice.h"
#include "isdBuffer.h"
#include "isdSource.h"
#include "isduStream.h"
#include "isduStreamSource.h"

namespace ist {
namespace isd {

SourceAdapter::SourceAdapter()
    : m_src(NULL)
{
}

SourceAdapter::~SourceAdapter()
{
    istSafeRelease(m_src);
}

void SourceAdapter::setSource(Source *v)
{
    istSafeAddRef(v);
    istSafeRelease(m_src);
    m_src = v;
}

Source* SourceAdapter::getSource()
{
    return m_src;
}

ALuint SourceAdapter::getHandle() const             { return m_src->getHandle(); }

bool SourceAdapter::isLooping() const               { return m_src->isLooping(); }
float SourceAdapter::getGain() const                { return m_src->getGain(); }
float SourceAdapter::getRefferenceDistance() const  { return m_src->getRefferenceDistance(); }
float SourceAdapter::getRolloffFactor() const       { return m_src->getRolloffFactor(); }
float SourceAdapter::getMaxDistance() const         { return m_src->getMaxDistance(); }
float SourceAdapter::getPitch() const               { return m_src->getPitch(); }
vec3 SourceAdapter::getPosition() const             { return m_src->getPosition(); }
vec3 SourceAdapter::getVelocity() const             { return m_src->getVelocity(); }
int SourceAdapter::getNumQueuedBuffers() const      { return m_src->getNumQueuedBuffers(); }
int SourceAdapter::getNumProcessedBuffers() const   { return m_src->getNumProcessedBuffers(); }

void SourceAdapter::setLooping(bool v)              { return m_src->setLooping(v); }
void SourceAdapter::setGain(float v)                { return m_src->setGain(v); }
void SourceAdapter::setRefferenceDistance(float v)  { return m_src->setRefferenceDistance(v); }
void SourceAdapter::setRolloffFactor(float v)       { return m_src->setRolloffFactor(v); }
void SourceAdapter::setMaxDistance(float v)         { return m_src->setMaxDistance(v); }
void SourceAdapter::setPitch(float v)               { return m_src->setPitch(v); }
void SourceAdapter::setPosition(const vec3& v)      { return m_src->setPosition(v); }
void SourceAdapter::setVelocity(const vec3& v)      { return m_src->setVelocity(v); }

bool SourceAdapter::isInitial() const               { return m_src->isLooping(); }
bool SourceAdapter::isPlaying() const               { return m_src->isLooping(); }
bool SourceAdapter::isPaused() const                { return m_src->isLooping(); }
bool SourceAdapter::isStopped() const               { return m_src->isLooping(); }

void SourceAdapter::play()                          { return m_src->play(); }
void SourceAdapter::pause()                         { return m_src->pause(); }
void SourceAdapter::stop()                          { return m_src->stop(); }
void SourceAdapter::rewind()                        { return m_src->rewind(); }

bool SourceAdapter::unqueue()                       { return m_src->unqueue(); }
void SourceAdapter::queue(Buffer *buf)              { return m_src->queue(buf); }
void SourceAdapter::clearQueue()                    { return m_src->clearQueue(); }



StreamSource::StreamSource(Device *dev) : m_stream(NULL), m_buffer_index(0)
{
    {
        Source *src = dev->createSource();
        setSource(src);
        istSafeRelease(src);
    }
    for(size_t i=0; i<_countof(m_internal_buffers); ++i) {
        m_internal_buffers[i] = dev->createBuffer();
    }
}

StreamSource::~StreamSource()
{
    for(size_t i=0; i<_countof(m_internal_buffers); ++i) {
        istSafeRelease(m_internal_buffers[i]);
    }
}

void StreamSource::setStream(Stream *ptr) { m_stream=ptr; fillBuffer(); }
Stream* StreamSource::getStream() { return m_stream; }

size_t StreamSource::size() { return m_stream->size(); }
size_t StreamSource::tell() { return m_stream->tell(); }
void StreamSource::seek(size_t v) { return m_stream->seek(v); }
bool StreamSource::eof() { return m_stream->eof(); }

void StreamSource::fillBuffer()
{
    for(int i=0; i<_countof(m_internal_buffers); ++i) {
        if(eof()) { break; }

        Buffer *buf = m_internal_buffers[i];
        DataCont& tmp = m_stream->readMillisec(250);
        buf->copy(&tmp[0], tmp.size(), m_stream->getALFormat(), m_stream->getSampleRate());
        queue(buf);
    }
}

void StreamSource::update()
{
    int processed = getNumProcessedBuffers();
    for(int i=0; i<processed; ++i) {
        if(eof()) { return; }

        DataCont& tmp = m_stream->readMillisec(250);
        unqueue();
        Buffer *buf = m_internal_buffers[m_buffer_index];
        buf->copy(&tmp[0], tmp.size(), m_stream->getALFormat(), m_stream->getSampleRate());
        queue(buf);

        m_buffer_index = (m_buffer_index+1) % s_num_buffers;
    }
}

} // namespace isd
} // namespace ist
