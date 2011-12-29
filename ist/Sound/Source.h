#ifndef __ist_Sound_Source__
#define __ist_Sound_Source__

namespace ist {
namespace sound {

    class Source
    {
    private:
        ALuint m_handle;

    protected:
        int getI(ALenum param) const;
        float getF(ALenum param) const;
        vec3 get3F(ALenum param) const;
        void setI(ALenum param, int v);
        void setF(ALenum param, float v);
        void set3F(ALenum param, const vec3& v);

    public:
        enum STATE {
            STATE_INITIAL   = AL_INITIAL,
            STATE_PLAYING   = AL_PLAYING,
            STATE_PAUSED    = AL_PAUSED,
            STATE_STOPPED   = AL_STOPPED,
        };

    public:
        Source();
        virtual ~Source();

        ALuint getHandle() const    { return m_handle; }

        int getBuffersQueued() const                { return getI(AL_BUFFERS_QUEUED); }
        bool getLooping(bool v) const               { return getI(AL_LOOPING)==AL_TRUE; }
        float getGain() const                       { return getF(AL_GAIN); }
        float getRefferenceDistance(float v) const  { return getF(AL_REFERENCE_DISTANCE); }
        float getMaxDistance(float v) const         { return getF(AL_MAX_DISTANCE); }
        float getRolloffFactor(float v) const       { return getF(AL_ROLLOFF_FACTOR); }
        vec3 getPosition() const                    { return get3F(AL_POSITION); }
        vec3 getVelocity() const                    { return get3F(AL_VELOCITY); }

        void setGain(float v)               { setF(AL_GAIN, v); }
        void setLooping(bool v)             { setI(AL_LOOPING, v); }
        void setRefferenceDistance(float v) { setF(AL_REFERENCE_DISTANCE, v); }
        void setMaxDistance(float v)        { setF(AL_MAX_DISTANCE, v); }
        void setRolloffFactor(float v)      { setF(AL_ROLLOFF_FACTOR, v); }
        void setPosition(const vec3& v)     { set3F(AL_POSITION, v); }
        void setVelocity(const vec3& v)     { set3F(AL_VELOCITY, v); }

        bool isInitial() const;
        bool isPlaying() const;
        bool isPaused() const;
        bool isStopped() const;
        int getProcessed() const;

        void play();
        void pause();
        void stop();
        void rewind();

        bool unqueue();
        void queue(Buffer *buf);
        void clearQueue();
    };
    typedef boost::shared_ptr<Source> SourcePtr;

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Source__
