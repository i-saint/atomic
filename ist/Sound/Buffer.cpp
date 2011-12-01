#include "stdafx.h"
#include "../Sound.h"

namespace ist {
namespace sound {

void Buffer::initialize()
{
    m_handle = 0;
    alGenBuffers(1, &m_handle);
}

int Buffer::getI(ALenum param) const
{
    int v = 0;
    alGetBufferi(m_handle, param, &v);
    return v;
}


Buffer::Buffer()
{
    initialize();
}

Buffer::Buffer(const std::string& filename)
{
    initialize();
    loadFromFile(filename);
}

Buffer::~Buffer()
{
    alDeleteBuffers(1, &m_handle);
}

ALuint Buffer::getHandle() const { return m_handle; }

int Buffer::getSize() const { return getI(AL_SIZE); }
int Buffer::getBits() const { return getI(AL_BITS); }
int Buffer::getChannels() const { return getI(AL_CHANNELS); }
int Buffer::getFrequency() const { return getI(AL_FREQUENCY); }


void Buffer::copy(char *data, size_t size, ALenum format, int samplerate)
{
    alBufferData(m_handle, format, data, size, samplerate);
}

/// .wavÇ©.oggÇéÊÇËçûÇﬁ 
bool Buffer::loadFromFile(const std::string& filename)
{
    //StreamPtr s(createStream(filename));
    //if(s) {
    //    std::vector<char>& tmp = s->readByte(s->size());
    //    copy(&tmp[0], tmp.size(), s->getALFormat(), s->getSampleRate());
    //    return true;
    //}
    return false;
}

} // namespace sound
} // namespace ist
