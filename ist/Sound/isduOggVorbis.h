#ifndef __ist_isd_OggVorbis__
#define __ist_isd_OggVorbis__

#include <vorbis/vorbisfile.h>
#include "isduOggVorbis.h"

namespace ist {
namespace isd {


// .ogg ファイルのストリーミング再生を実現。
class OggVorbisFileStream : public Stream
{
private:
    FILE *m_file;
    OggVorbis_File m_ov;
    vorbis_info *m_info;
    DataCont m_tmp;

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

    DataCont& readByte(size_t require_size);
};

// .ogg ファイルの内容をメモリに配置して、それをストリーミング再生する機能を実現
class OggVorbisMemoryStream : public Stream
{
private:
    void *m_data;
    OggVorbis_File m_ov;
    vorbis_info *m_info;
    size_t m_datasize;
    size_t m_position;
    DataCont m_tmp;

public:
    OggVorbisMemoryStream();
    virtual ~OggVorbisMemoryStream();

    // data はコピーせず共有する
    bool openStream(void* data, size_t data_size);
    void closeStream();

    size_t size();
    size_t tell();
    void seek(size_t v);
    bool eof();

    size_t getChannels() const;
    size_t getSampleRate() const;
    size_t getBitsPerSample() const;

    DataCont& readByte(size_t require_size);

    // callbacks called by ogg vorbis
    static size_t read( void* buffer, size_t size, size_t maxCount, void* stream );
    static int seek( void* buffer, ogg_int64_t offset, int flag );
    static int close( void* buffer );
    static long tell( void* buffer );
};


} // namespace isd
} // namespace ist

#endif // __ist_isd_OggVorbis__
