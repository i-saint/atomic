#include "stdafx.h"
#include "../Sound.h"
#include "../Base.h"

#ifdef IST_SOUND_ENABLE_OGGVORBIS

namespace ist {
namespace sound {

OggVorbisFileStream::OggVorbisFileStream()
: m_file(NULL)
{
    m_tmp.reserve(4096);
}

OggVorbisFileStream::~OggVorbisFileStream()
{
    ov_clear(&m_ov);
}


bool OggVorbisFileStream::openStream(const char* filepath)
{
    m_file = fopen(filepath, "rb");
    if(!m_file) {
        IST_PRINT("OggVorbisStream::OggVorbisStream(): file not found %s\n", filepath);
        goto LABEL_ERROR;
    }
    int e = ov_open(m_file, &m_ov, 0,0);
    if(e!=0) {
        IST_PRINT("OggVorbisStream::OggVorbisStream(): not vorbis file %s\n", filepath);
        goto LABEL_ERROR;
    }
    m_info = ov_info(&m_ov, -1);
    return true;

LABEL_ERROR:
    closeStream();
    return false;
}

void OggVorbisFileStream::closeStream()
{
    if(m_file) { fclose(m_file); m_file=NULL; }
}


size_t OggVorbisFileStream::size()
{
    return ov_pcm_total(&m_ov, -1)*getChannels()*(getBitsPerSample()/8);
}

size_t OggVorbisFileStream::tell()
{
    return ov_pcm_tell(&m_ov)*getChannels()*(getBitsPerSample()/8);
}

void OggVorbisFileStream::seek(size_t v)
{
    ov_pcm_seek(&m_ov, v);
}

bool OggVorbisFileStream::eof()
{
    return m_file==NULL || tell()==size();
}

size_t OggVorbisFileStream::getChannels() const { return m_info->channels; }
size_t OggVorbisFileStream::getSampleRate() const { return m_info->rate; }
size_t OggVorbisFileStream::getBitsPerSample() const { return 16; }

OggVorbisFileStream::CharCont& OggVorbisFileStream::readByte(size_t require_size)
{
    require_size -= require_size % (getChannels()*2); // alignment 
    m_tmp.clear();
    char tmp[4096];
    int bitstream = 0;
    for(;;) {
        long read = ov_read(&m_ov, tmp, std::min<size_t>(4096, require_size-m_tmp.size()), 0, 2, 1, &bitstream);
        m_tmp.insert(m_tmp.end(), tmp, tmp+read);
        if(read<=0 || m_tmp.size()>=require_size) {
            break;
        }
    }
    return m_tmp;
}





OggVorbisMemoryStream::OggVorbisMemoryStream()
: m_data(NULL)
, m_info(NULL)
, m_datasize(0)
, m_position(0)
{
}

OggVorbisMemoryStream::~OggVorbisMemoryStream()
{
    closeStream();
}


bool OggVorbisMemoryStream::openStream(void* data)
{
    closeStream();
    m_data = data;


    ov_callbacks callbacks = {
        &OggVorbisMemoryStream::read,
        &OggVorbisMemoryStream::seek,
        &OggVorbisMemoryStream::close,
        &OggVorbisMemoryStream::tell
    };
    if ( ov_open_callbacks( data, &m_ov , 0, 0, callbacks ) != 0 ) {
        closeStream();
        return false;
    }

    m_info = ov_info( &m_ov, -1 );

    return true;
}

void OggVorbisMemoryStream::closeStream()
{
    m_data = NULL;
    m_info = NULL;
    m_datasize = 0;
    m_position = 0;
}


size_t OggVorbisMemoryStream::size()
{
    return ov_pcm_total(&m_ov, -1)*getChannels()*(getBitsPerSample()/8);
}

size_t OggVorbisMemoryStream::tell()
{
    return ov_pcm_tell(&m_ov)*getChannels()*(getBitsPerSample()/8);
}

void OggVorbisMemoryStream::seek(size_t v)
{
    ov_pcm_seek(&m_ov, v);
}

bool OggVorbisMemoryStream::eof()
{
    return m_data==NULL || tell()==size();
}
size_t OggVorbisMemoryStream::getChannels() const { return m_info->channels; }
size_t OggVorbisMemoryStream::getSampleRate() const { return m_info->rate; }
size_t OggVorbisMemoryStream::getBitsPerSample() const { return 16; }



size_t OggVorbisMemoryStream::read( void* buffer, size_t size, size_t maxCount, void* stream )
{
    if(buffer==NULL) {
        return 0;
    }

    OggVorbisMemoryStream *p = static_cast<OggVorbisMemoryStream*>(stream);

    int resSize = p->m_datasize - p->m_position;
    size_t count = resSize / size;
    if ( count > maxCount ) {
        count = maxCount;
    }

    memcpy( buffer, static_cast<char*>(p->m_data) + p->m_position, size*count );

    p->m_position += size * count;

    return count;
}

int OggVorbisMemoryStream::seek( void* stream, ogg_int64_t offset, int flag )
{
    OggVorbisMemoryStream *p = static_cast<OggVorbisMemoryStream*>(stream);

    switch( flag ) {
    case SEEK_CUR:
        p->m_position += offset;
        break;

    case SEEK_END:
        p->m_position = p->m_datasize + offset;
        break;

    case SEEK_SET:
        p->m_position = offset;
        break;

    default:
        return -1;
    }

    if ( p->m_position > p->m_datasize ) {
        p->m_position = p->m_datasize;
        return -1;
    }
    else if ( p->m_position < 0 ) {
        p->m_position = 0;
        return -1;
    }

    return 0;
}

int OggVorbisMemoryStream::close( void* stream )
{
    return 0;
}

long OggVorbisMemoryStream::tell( void* stream )
{
    return static_cast<OggVorbisMemoryStream*>(stream)->m_position;
}

} // namespace sound
} // namespace ist

#endif // IST_SOUND_ENABLE_OGGVORBIS
