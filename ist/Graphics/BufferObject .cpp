#include "stdafx.h"
#include "GraphicsAssert.h"
#include "BufferObject.h"

namespace ist {
namespace graphics {

template<GLuint BufferType>
BufferObject<BufferType>::BufferObject() : m_buffer_object(0), m_size(0)
{
    glGenBuffers(1, &m_buffer_object);
    CheckGLError();
}

template<GLuint BufferType>
BufferObject<BufferType>::~BufferObject()
{
    glDeleteBuffers(1, &m_buffer_object);
    CheckGLError();
}

template<GLuint BufferType>
GLuint BufferObject<BufferType>::size() const
{
    return m_size;
}

template<GLuint BufferType>
void BufferObject<BufferType>::bind() const
{
    glBindBuffer(BufferType, m_buffer_object);
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
    glBindBuffer(BufferType, m_buffer_object);
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
    glBindBuffer(BufferType, m_buffer_object);
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
    glBindBuffer(BufferType, m_buffer_object);
    glBufferData(BufferType, size, data, usage);
    CheckGLError();
    glBindBuffer(BufferType, 0);
}

template BufferObject<GL_ARRAY_BUFFER>;
template BufferObject<GL_ELEMENT_ARRAY_BUFFER>;
template BufferObject<GL_PIXEL_PACK_BUFFER>;
template BufferObject<GL_PIXEL_UNPACK_BUFFER>;
template BufferObject<GL_UNIFORM_BUFFER>;


} // namespace graphics
} // namespace ist
