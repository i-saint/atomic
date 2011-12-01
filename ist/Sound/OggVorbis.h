#ifndef __ist_Sound_OggVorbis__
#define __ist_Sound_OggVorbis__

namespace ist {
namespace sound {


class OggVorbisFileStream : public Stream
{
private:
    FILE *m_file;
    OggVorbis_File m_ov;
    vorbis_info *m_info;
    CharCont m_tmp;

public:
    OggVorbisFileStream();
    virtual ~OggVorbisFileStream();

    bool openStream(const char* filepath);
    void closeStream();

    size_t size();
    size_t tell();
    void seek(size_t v);
    bool eof();

    size_t getChannels() const;
    size_t getSampleRate() const;
    size_t getBitsPerSample() const;

    CharCont& readByte(size_t require_size);
};

class OggVorbisMemoryStream : public Stream
{
private:
    void *m_data;
    OggVorbis_File m_ov;
    vorbis_info *m_info;
    size_t m_datasize;
    size_t m_position;

public:
    OggVorbisMemoryStream();
    virtual ~OggVorbisMemoryStream();

    bool openStream(void* data);
    void closeStream();

    size_t size();
    size_t tell();
    void seek(size_t v);
    bool eof();

    size_t getChannels() const;
    size_t getSampleRate() const;
    size_t getBitsPerSample() const;

    // callbacks called by ogg vorbis
    static size_t read( void* buffer, size_t size, size_t maxCount, void* stream );
    static int seek( void* buffer, ogg_int64_t offset, int flag );
    static int close( void* buffer );
    static long tell( void* buffer );
};


} // namespace sound
} // namespace ist

#endif //__ist_Sound_OggVorbis__
