#include "istPCH.h"
#include "../Sound.h"

namespace ist {
namespace isd {

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

Buffer::~Buffer()
{
    alDeleteBuffers(1, &m_handle);
}

int Buffer::getSize() const { return getI(AL_SIZE); }
int Buffer::getBits() const { return getI(AL_BITS); }
int Buffer::getChannels() const { return getI(AL_CHANNELS); }
int Buffer::getFrequency() const { return getI(AL_FREQUENCY); }


void Buffer::copy(char *data, size_t size, ALenum format, int samplerate)
{
    alBufferData(m_handle, format, data, size, samplerate);
}

} // namespace isd
} // namespace ist
