#include "stdafx.h"
#include "../Base.h"
#include "i3dBuffer.h"

namespace ist {
namespace i3d {

template<GLuint BufferType>
Buffer<BufferType>::Buffer()
: m_size(0)
, m_capacity(0)
{
    glGenBuffers(1, &m_handle);
}

template<GLuint BufferType>
Buffer<BufferType>::~Buffer()
{
    if(m_handle!=0) {
        glDeleteBuffers(1, &m_handle);
        m_handle = 0;
    }
}

template<GLuint BufferType>
void Buffer<BufferType>::allocate(GLuint size, I3D_USAGE usage, void *data)
{
    m_size = size;
    if(size==0) {
        return;
    }
    else if(size > m_capacity) {
        m_capacity = size;
        glBindBuffer(BufferType, m_handle);
        glBufferData(BufferType, size, data, usage);
        glBindBuffer(BufferType, 0);
    }
    else if(data!=NULL) {
        glBindBuffer(BufferType, m_handle);
        void *p = glMapBuffer(BufferType, GL_WRITE_ONLY);
        memcpy(p, data, size);
        glUnmapBuffer(BufferType);
        glBindBuffer(BufferType, 0);
    }
}

template<GLuint BufferType>
GLuint Buffer<BufferType>::size() const
{
    return m_size;
}

template<GLuint BufferType>
void Buffer<BufferType>::bind() const
{
    glBindBuffer(BufferType, m_handle);
}

template<GLuint BufferType>
void Buffer<BufferType>::unbind() const
{
    glBindBuffer(BufferType, 0);
}

template<GLuint BufferType>
void* Buffer<BufferType>::map(I3D_MAP_MODE mode)
{
    glBindBuffer(BufferType, m_handle);
    void *r = glMapBuffer(BufferType, mode);
    if(r==NULL) { istAssert("BufferObject::map() failed\n"); }
    glBindBuffer(BufferType, 0);
    return r;
}

template<GLuint BufferType>
void Buffer<BufferType>::unmap()
{
    glBindBuffer(BufferType, m_handle);
    glUnmapBuffer(BufferType);
    glBindBuffer(BufferType, 0);
}

template Buffer<GL_ARRAY_BUFFER>;
template Buffer<GL_ELEMENT_ARRAY_BUFFER>;
template Buffer<GL_PIXEL_PACK_BUFFER>;
template Buffer<GL_PIXEL_UNPACK_BUFFER>;
template Buffer<GL_UNIFORM_BUFFER>;


void UniformBuffer::bindBase(GLuint index) const
{
    glBindBufferBase(GL_UNIFORM_BUFFER, index, getHandle());
}

void UniformBuffer::bindRange(GLuint index, GLintptr offset, GLsizeiptr size) const
{
    glBindBufferRange(GL_UNIFORM_BUFFER, index, getHandle(), offset, size);
}


VertexArray::VertexArray()
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

void VertexArray::setAttributes( VertexBuffer& vbo, size_t stride, const VertexDescriptor *descs, size_t num_descs )
{
    glBindVertexArray(m_handle);
    vbo.bind();
    for(size_t i=0; i<num_descs; ++i) {
        const VertexDescriptor& desc = descs[i];
        glEnableVertexAttribArray(desc.location);
        glVertexAttribPointer(desc.location, desc.num_elements, desc.type, desc.normalize, stride, (GLvoid*)desc.offset);
        glVertexAttribDivisor(desc.location, desc.divisor);
    }
    vbo.unbind();
}


} // namespace i3d
} // namespace ist
