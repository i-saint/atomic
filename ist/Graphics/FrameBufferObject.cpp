#include "stdafx.h"
#include "../Base/Assert.h"
#include "FrameBufferObject.h"

namespace ist {
namespace graphics {

Texture2D::Texture2D()
{
    glGenTextures(1, &m_handle);
}

Texture2D::~Texture2D()
{
    glDeleteTextures(1, &m_handle);
}

bool Texture2D::initialize(GLsizei width, GLsizei height, FORMAT fmt, void *data)
{
    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    switch(fmt)
    {
    case FMT_RGBA_I8:
        internal_format = GL_RGBA;
        format = GL_RGBA;
        type = GL_UNSIGNED_BYTE;
        break;
    case FMT_DEPTH_F32:
        internal_format = GL_DEPTH_COMPONENT32;
        format = GL_DEPTH_COMPONENT;
        type = GL_FLOAT;
        break;
    default:
        IST_ASSERT("unknown format: %d", fmt);
        return false;
    }

    glBindTexture( GL_TEXTURE_2D, m_handle );
    glTexImage2D( GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, data );
    glBindTexture( GL_TEXTURE_2D, 0 );
    return true;
}

void Texture2D::bind() const
{
    glBindTexture(GL_TEXTURE_2D, m_handle);
}
void Texture2D::unbind() const
{
    glBindTexture(GL_TEXTURE_2D, m_handle);
}



RenderBuffer::RenderBuffer()
{
    glGenRenderbuffers(1, &m_handle);
}

RenderBuffer::~RenderBuffer()
{
    glDeleteRenderbuffers(1, &m_handle);
}

bool RenderBuffer::initialize(GLsizei width, GLsizei height, FORMAT fmt)
{
    GLint internal_format = 0;
    switch(fmt)
    {
    case FMT_RGBA_I8:
        internal_format = GL_RGBA;
        break;
    case FMT_DEPTH_F32:
        internal_format = GL_DEPTH_COMPONENT32;
        break;
    default:
        IST_ASSERT("unknown format: %d", fmt);
        return false;
    }

    glBindRenderbuffer( GL_RENDERBUFFER, m_handle );
    glRenderbufferStorage( GL_RENDERBUFFER, internal_format, width, height );
    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
    return true;
}

void RenderBuffer::bind() const
{
    glBindRenderbuffer( GL_RENDERBUFFER, m_handle );
}

void RenderBuffer::unbind() const
{
    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
}



FrameBufferObject::FrameBufferObject()
{
    glGenFramebuffers(1, &m_handle);
}

FrameBufferObject::~FrameBufferObject()
{
    glDeleteFramebuffers(1, &m_handle);
}

bool FrameBufferObject::attachRenderBuffer(RenderBuffer& rb, ATTACH attach)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attach, GL_RENDERBUFFER, rb.getHandle());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

bool FrameBufferObject::attachTexture(Texture2D& tex, ATTACH attach, GLint level)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attach, GL_TEXTURE_2D, tex.getHandle(), level);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void FrameBufferObject::bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
}

void FrameBufferObject::unbind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace graphics
} // namespace ist


