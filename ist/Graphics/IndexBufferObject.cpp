#include "stdafx.h"
#include "GraphicsAssert.h"
#include "IndexBufferObject.h"

namespace ist {
namespace graphics {


IndexBufferObject::IndexBufferObject() : m_buffer_object(0), m_size(0)
{
    glGenBuffers(1, &m_buffer_object);
    CheckGLError();
}

IndexBufferObject::~IndexBufferObject()
{
    glDeleteBuffers(1, &m_buffer_object);
    CheckGLError();
}

GLuint IndexBufferObject::size() const
{
    return m_size;
}

void IndexBufferObject::bind() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer_object);
}

void IndexBufferObject::unbind() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void* IndexBufferObject::lock(LOCK mode)
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer_object);
    void *r = glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, mode);
    CheckGLError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    return r;
}

void IndexBufferObject::unlock()
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer_object);
    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    CheckGLError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void IndexBufferObject::allocate(GLuint size, USAGE usage, void *data)
{
    m_size = size;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer_object);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, usage);
    CheckGLError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

} // namespace graphics
} // namespace ist
