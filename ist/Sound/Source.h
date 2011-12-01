#ifndef __ist_Sound_Source__
#define __ist_Sound_Source__

namespace ist {
namespace sound {

    class Source
    {
    private:
        typedef std::deque<BufferPtr> BufferQueue;
        ALuint m_handle;
        BufferQueue m_queue;

    protected:
        int getI(ALenum param) const;
        float getF(ALenum param) const;
        float4 get3F(ALenum param) const;
        void setI(ALenum param, int v);
        void setF(ALenum param, float v);
        void set3F(ALenum param, const float4& v);

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

        float getGain() const;
        float4 getPosition() const;
        float4 getVelocity() const;

        void setGain(float v);
        void setPosition(const float4& v);
        void setVelocity(const float4& v);

        bool isInitial() const;
        bool isPlaying() const;
        bool isPaused() const;
        bool isStopped() const;
        int getProcessed() const;

        void play();
        void pause();
        void stop();
        void rewind();

        BufferPtr unqueue();
        void queue(BufferPtr buf);

        virtual void update();
    };
    typedef boost::shared_ptr<Source> SourcePtr;

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Source__
