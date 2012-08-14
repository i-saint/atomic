#include "istPCH.h"
#include "../Base.h"
#include "../Sound.h"

namespace ist {
namespace isd {

Stream::~Stream() {}

ALenum Stream::getALFormat() const
{
    if(getChannels()==1) {
        if(getBitsPerSample()==8) {
            return AL_FORMAT_MONO8;
        }
        else if(getBitsPerSample()==16) {
            return AL_FORMAT_MONO16;
        }
    }
    else if(getChannels()==2) {
        if(getBitsPerSample()==8) {
            return AL_FORMAT_STEREO8;
        }
        else if(getBitsPerSample()==16) {
            return AL_FORMAT_STEREO16;
        }
    }
    else if(getChannels()==4) {
        return alGetEnumValue("AL_FORMAT_QUAD16");
    }
    else if(getChannels()==6) {
        return alGetEnumValue("AL_FORMAT_51CHN16");
    }
    return (ALenum)0;
}

Stream::DataCont& Stream::readMillisec(size_t require_millisec)
{
    size_t require_size = size_t(
        float(getSampleRate()*getChannels()*(getBitsPerSample()/8)) * float(require_millisec)/1000.0f);
    return readByte(require_size);
}

Stream::DataCont& Stream::readSample(size_t require_sample)
{
    size_t require_size = require_sample*getChannels()*(getBitsPerSample()/8);
    return readByte(require_size);
}




WaveStream::WaveStream()
: m_file(NULL)
{
    m_tmp.reserve(4096);
}

WaveStream::~WaveStream()
{
    closeStream();
}

bool WaveStream::openStream(const char* filepath)
{
    closeStream();

    m_file = fopen(filepath, "rb");
    if(!m_file) {
        return false;
    }

    fread(&m_riff, sizeof(riff_header), 1, m_file);
    if(strncmp(m_riff.riff, "RIFF", 4)!=0) {
        istPrint("WaveStream::WaveStream(): not RIFF file %s\n", filepath);
        goto section_error;
    }
    if(strncmp(m_riff.wave, "WAVE", 4)!=0){
        istPrint("WaveStream::WaveStream(): not WAVE file %s\n", filepath);
        goto section_error;
    }

    fread(&m_format, sizeof(format_header), 1, m_file);
    if(strncmp(m_format.fmt, "fmt ", 4)!=0){
        istPrint("WaveStream::WaveStream(): fmt not found %s\n", filepath);
        goto section_error;
    }

    fread(&m_data, sizeof(data_header), 1, m_file);
    if(strncmp(m_data.data, "data", 4)!=0){
        istPrint("WaveStream::WaveStream(): data not found %s\n", filepath);
        goto section_error;
    }

    return true;

section_error:
    closeStream();
    return false;
}

void WaveStream::closeStream()
{
    if(m_file) { fclose(m_file); m_file=NULL; }
}

size_t WaveStream::getHeaderSize() const { return sizeof(riff_header)+sizeof(format_header)+sizeof(data_header); }

size_t WaveStream::size()       { return m_data.size; }
size_t WaveStream::tell()       { return ftell(m_file)-getHeaderSize(); }
void WaveStream::seek(size_t v) { fseek(m_file, getHeaderSize()+v, SEEK_SET); }
bool WaveStream::eof()          { return m_file==NULL || tell()==size(); }

size_t WaveStream::getChannels() const      { return m_format.channels; }
size_t WaveStream::getSampleRate() const    { return m_format.sample_rate; }
size_t WaveStream::getBitsPerSample() const { return m_format.bits_per_sample; }

WaveStream::DataCont& WaveStream::readByte(size_t require_size)
{
    m_tmp.clear();
    m_tmp.resize(stl::min<size_t>(require_size, size()-tell()));
    fread(&m_tmp[0], 1, m_tmp.size(), m_file);
    return m_tmp;
}

} // namespace isd
} // namespace ist
