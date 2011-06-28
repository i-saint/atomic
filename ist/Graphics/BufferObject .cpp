#include "stdafx.h"
#include "GraphicsAssert.h"
#include "BufferObject.h"

namespace ist {
namespace graphics {

template<GLuint BufferType>
BufferObject<BufferType>::BufferObject()
: m_handle(0)
, m_size(0)
{
}

template<GLuint BufferType>
BufferObject<BufferType>::~BufferObject()
{
    finalize();
}


template<GLuint BufferType>
void BufferObject<BufferType>::initialize()
{
    glGenBuffers(1, &m_handle);
    CheckGLError();
}

template<GLuint BufferType>
void BufferObject<BufferType>::finalize()
{
    if(m_handle!=0) {
        glDeleteBuffers(1, &m_handle);
        CheckGLError();
    }
    m_handle = 0;
}

template<GLuint BufferType>
GLuint BufferObject<BufferType>::size() const
{
    return m_size;
}

template<GLuint BufferType>
void BufferObject<BufferType>::bind() const
{
    glBindBuffer(BufferType, m_handle);
    CheckGLError();
}

template<GLuint BufferType>
void BufferObject<BufferType>::unbind() const
{
    glBindBuffer(BufferType, 0);
    CheckGLError();
}

template<GLuint BufferType>
void* BufferObject<BufferType>::lock(LOCK mode)
{
    glBindBuffer(BufferType, m_handle);
    CheckGLError();
    void *r = glMapBuffer(BufferType, mode);
    CheckGLError();
    glBindBuffer(BufferType, 0);
    CheckGLError();
    return r;
}

template<GLuint BufferType>
void BufferObject<BufferType>::unlock()
{
    glBindBuffer(BufferType, m_handle);
    CheckGLError();
    glUnmapBuffer(BufferType);
    CheckGLError();
    glBindBuffer(BufferType, 0);
    CheckGLError();
}

template<GLuint BufferType>
void BufferObject<BufferType>::allocate(GLuint size, USAGE usage, void *data)
{
    m_size = size;
    glBindBuffer(BufferType, m_handle);
    glBufferData(BufferType, size, data, usage);
    CheckGLError();
    glBindBuffer(BufferType, 0);
}

template BufferObject<GL_ARRAY_BUFFER>;
template BufferObject<GL_ELEMENT_ARRAY_BUFFER>;
template BufferObject<GL_PIXEL_PACK_BUFFER>;
template BufferObject<GL_PIXEL_UNPACK_BUFFER>;
template BufferObject<GL_UNIFORM_BUFFER>;


void UniformBufferObject::bindBase(GLuint index) const
{
    glBindBufferBase(GL_UNIFORM_BUFFER, index, getHandle());
}

void UniformBufferObject::bindRange(GLuint index, GLintptr offset, GLsizeiptr size) const
{
    glBindBufferRange(GL_UNIFORM_BUFFER, index, getHandle(), offset, size);
}


} // namespace graphics
} // namespace ist
