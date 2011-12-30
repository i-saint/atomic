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

        ALuint getHandle() const;

        bool isLooping(bool v) const;
        float getGain() const;
        float getRefferenceDistance(float v) const;
        float getRolloffFactor(float v) const;
        float getMaxDistance(float v) const;
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
    typedef boost::shared_ptr<Source> SourcePtr;

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Source__
