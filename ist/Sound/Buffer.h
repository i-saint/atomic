#ifndef __ist_Sound_Buffer__
#define __ist_Sound_Buffer__

namespace ist {
namespace sound {

    class Buffer
    {
    private:
        ALuint m_handle;

    private:
        void initialize();

    protected:
        int getI(ALenum param) const;

    public:
        Buffer();
        virtual ~Buffer();
        ALuint getHandle() const;

        int getSize() const;
        int getBits() const;
        int getChannels() const;
        int getFrequency() const;

        /// format: AL_FORMAT_MONO8  AL_FORMAT_MONO16 AL_FORMAT_STEREO8 AL_FORMAT_STEREO16 
        void copy(char *data, size_t size, ALenum format, int samplerate);
    };

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Buffer__
