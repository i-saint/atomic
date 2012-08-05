#include "stdafx.h"
#include "../Base.h"
#include "i3dglBuffer.h"

namespace ist {
namespace i3dgl {

Buffer::Buffer(Device *dev, const BufferDesc &desc)
: super(dev)
, m_desc(desc)
{
    glGenBuffers(1, &m_handle);

    glBindBuffer(m_desc.type, m_handle);
    glBufferData(m_desc.type, m_desc.size, m_desc.data, m_desc.usage);
    glBindBuffer(m_desc.type, 0);
}

Buffer::~Buffer()
{
    if(m_handle!=0) {
        glDeleteBuffers(1, &m_handle);
        m_handle = 0;
    }
}

void Buffer::bind() const
{
    glBindBuffer(m_desc.type, m_handle);
}

void Buffer::unbind() const
{
    glBindBuffer(m_desc.type, 0);
}

void* Buffer::map(I3D_MAP_MODE mode)
{
    glBindBuffer(m_desc.type, m_handle);
    void *r = glMapBuffer(m_desc.type, mode);
    istAssert(r!=NULL, "BufferObject::map() failed\n");
    glBindBuffer(m_desc.type, 0);
    return r;
}

void Buffer::unmap()
{
    glBindBuffer(m_desc.type, m_handle);
    glUnmapBuffer(m_desc.type);
    glBindBuffer(m_desc.type, 0);
}



VertexArray::VertexArray(Device *dev)
    : super(dev)
{
    glGenVertexArrays(1, &m_handle);
}

VertexArray::~VertexArray()
{
    if(m_handle!=0) {
        glDeleteVertexArrays(1, &m_handle);
        m_handle = 0;
    }
}

void VertexArray::bind() const
{
    glBindVertexArray(m_handle);
}

void VertexArray::unbind() const
{
    glBindVertexArray(0);
}

void VertexArray::setAttributes( Buffer& vb, size_t stride, const VertexDesc *descs, size_t num_descs )
{
    glBindVertexArray(m_handle);
    vb.bind();
    for(size_t i=0; i<num_descs; ++i) {
        const VertexDesc& desc = descs[i];
        glEnableVertexAttribArray(desc.location);
        // float type
        if(desc.type==I3D_HALF || desc.type==I3D_FLOAT || desc.type==I3D_DOUBLE) {
            glVertexAttribPointer(desc.location, desc.num_elements, desc.type, desc.normalize, stride, (GLvoid*)desc.offset);
        }
        // integer type
        else {
            glVertexAttribIPointer(desc.location, desc.num_elements, desc.type, stride, (GLvoid*)desc.offset);
        }
        glVertexAttribDivisor(desc.location, desc.divisor);
    }
    vb.unbind();
}


} // namespace i3d
} // namespace ist
