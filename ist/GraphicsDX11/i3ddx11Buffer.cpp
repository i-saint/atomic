#include "stdafx.h"
#include "../Base.h"
#include "i3ddx11Buffer.h"

namespace ist {
namespace i3ddx11 {

Buffer::Buffer(Device *dev)
: DeviceResource(dev)
, m_size(0)
, m_capacity(0)
{
}

Buffer::~Buffer()
{
}

void Buffer::allocate(GLuint size, I3D_USAGE usage, void *data)
{
}

GLuint Buffer::size() const
{
    return m_size;
}

void Buffer::bind() const
{
}

void Buffer::unbind() const
{
}

void* Buffer::map(I3D_MAP_MODE mode)
{
}

void Buffer::unmap()
{
}




} // namespace i3ddx11
} // namespace ist
