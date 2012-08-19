#ifndef __ist_isd_SourceSet__
#define __ist_isd_SourceSet__

#include "isdTypes.h"

namespace ist {
namespace isd {

class istInterModule SourceAdapter : public SharedObject
{
private:
    Source *m_src;

protected:
    SourceAdapter();
    ~SourceAdapter();

public:
    void setSource(Source *v);
    Source* getSource();

    ALuint getHandle() const;

    bool isLooping() const;
    float getGain() const;
    float getRefferenceDistance() const;
    float getRolloffFactor() const;
    float getMaxDistance() const;
    float getPitch() const;
    vec3 getPosition() const;
    vec3 getVelocity() const;
    int getNumQueuedBuffers() const;
    int getNumProcessedBuffers() const;

    void setLooping(bool v);
    void setGain(float v);
    void setRefferenceDistance(float v);
    void setRolloffFactor(float v);
    void setMaxDistance(float v);
    void setPitch(float v);             // 0.0f - 1.0f. default: 1.0f
    void setPosition(const vec3& v);
    void setVelocity(const vec3& v);

    bool isInitial() const;
    bool isPlaying() const;
    bool isPaused() const;
    bool isStopped() const;

    void play();
    void pause();
    void stop();
    void rewind();

    bool unqueue();
    void queue(Buffer *buf);
    void clearQueue();
};


class istInterModule StreamSource : public SourceAdapter
{
public:
    typedef stl::vector<char> DataCont;

private:
    static const int s_num_buffers = 4;
    Stream *m_stream; // shared
    Buffer *m_internal_buffers[s_num_buffers];  // internal
    int m_buffer_index;

public:
    StreamSource(Device *dev);
    virtual ~StreamSource();

    void setStream(Stream *ptr);
    Stream* getStream();

    size_t size();
    size_t tell();
    void seek(size_t v);
    bool eof();

    void fillBuffer();
    void update();
};


template<class SourceType>
class istInterModule TSourceSet : public SharedObject
{
public:
    typedef stl::vector<SourceType*> SourceCont;
    typedef stl::vector<ALuint> HandleCont;
private:
    SourceCont m_sources;
    HandleCont m_handles;

public:
    TSourceSet()
    {}

    virtual ~TSourceSet()
    {
        for(uint32 i=0; i<m_sources.size(); ++i) {
            istSafeRelease(m_sources[i]);
        }
    }

    virtual void append(SourceType *v)
    {
        m_sources.push_back(v);
        gatherHandle();
    }

    virtual void gatherHandle()
    {
        m_handles.clear();
        for(size_t i=0; i<m_sources.size(); ++i) {
            m_handles.push_back(m_sources[i]->getHandle());
        }
    }

    virtual size_t getSourceSize() const { return m_sources.size(); }
    virtual SourceType* getSource(size_t i) { return m_sources[i]; }


    virtual void setGain(float v)
    {
        for(size_t i=0; i<getSourceSize(); ++i) {
            getSource(i)->setGain(v);
        }
    }


    virtual bool isPlaying()
    {
        for(size_t i=0; i<getSourceSize(); ++i) {
            if(getSource(i)->isPlaying()) {
                return true;
            }
        }
        return false;
    }

    virtual void play()
    {
        for(size_t i=0; i<getSourceSize(); ++i) {
            getSource(i)->play();
        }
    }

    virtual void update()
    {
        for(size_t i=0; i<getSourceSize(); ++i) {
            getSource(i)->update();
        }
    }
};
typedef TSourceSet<Source> SourceSet;

class istInterModule StreamSourceSet : public TSourceSet<StreamSource>
{
public:

    void seek(size_t size)
    {
        for(size_t i=0; i<getSourceSize(); ++i) {
            getSource(i)->seek(size);
        }
    }

    bool eof()
    {
        for(size_t i=0; i<getSourceSize(); ++i) {
            if(!getSource(i)->eof()) {
                return false;
            }
        }
        return true;
    }
};


} // namespace isd
} // namespace ist

#endif // __ist_isd_SourceSet__
