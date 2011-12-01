#ifndef __ist_Sound_Stream__
#define __ist_Sound_Stream__

namespace ist {
namespace sound {

    class Stream
    {
    public:
        typedef std::vector<char> CharCont;

    public:
        virtual ~Stream();

        virtual size_t getChannels() const=0;
        virtual size_t getSampleRate() const=0;
        virtual size_t getBitsPerSample() const=0;

        virtual size_t size()=0;
        virtual size_t tell()=0;
        virtual void seek(size_t size)=0;
        virtual bool eof()=0;

        virtual ALenum getALFormat() const;

        /// require_size byte ï™ì«Ç›çûÇﬁ 
        virtual CharCont& readByte(size_t require_size)=0;

        /// require_millisecÉ~Éäïbï™ì«Ç›çûÇﬁ 
        CharCont& readMillisec(size_t require_millisec);

        /// require_sampleÉTÉìÉvÉãï™ì«Ç›çûÇﬁ 
        CharCont& readSample(size_t require_sample);
    };
    typedef boost::shared_ptr<Stream> StreamPtr;




    class WaveStream : public Stream
    {
    public:
        struct riff_header {
            char riff[4];
            int size;
            char wave[4];
        };

        struct format_header {
            char fmt[4];
            int size;
            short format_id;
            short channels;
            int sample_rate;
            int bytes_par_second;
            short block_size;
            short bits_per_sample;
        };

        struct data_header {
            char data[4];
            int size;
        };

    private:
        FILE *m_file;
        riff_header m_riff;
        format_header m_format;
        data_header m_data;
        CharCont m_tmp;

    public:
        WaveStream();
        ~WaveStream();

        bool openStream(const char* path);
        void closeStream();

        size_t getHeaderSize() const;

        size_t size();
        size_t tell();
        void seek(size_t v);
        bool eof();

        size_t getChannels() const;
        size_t getSampleRate() const;
        size_t getBitsPerSample() const;

        CharCont& readByte(size_t require_size);
    };

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Stream__
