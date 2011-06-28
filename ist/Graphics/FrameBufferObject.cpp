#include "stdafx.h"
#include "../Base/Assert.h"
#include "FrameBufferObject.h"

namespace ist {
namespace graphics {

Texture2D::Texture2D()
: m_handle(0)
, m_width(0)
, m_height(0)
{
}

Texture2D::~Texture2D()
{
    finalize();
}

bool Texture2D::initialize()
{
    glGenTextures(1, &m_handle);
    return true;
}

bool Texture2D::initialize(GLsizei width, GLsizei height, FORMAT fmt, void *data)
{
    return initialize() && allocate(width, height, fmt, data);
}

void Texture2D::finalize()
{
    if(m_handle!=0) {
        glDeleteTextures(1, &m_handle);
    }
    m_handle = 0;
}

bool Texture2D::allocate(GLsizei width, GLsizei height, FORMAT fmt, void *data)
{
    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    switch(fmt)
    {
    case FMT_RGB_U8:
        internal_format = GL_RGB8;
        format = GL_RGB;
        type = GL_UNSIGNED_BYTE;
        break;
    case FMT_RGBA_U8:
        internal_format = GL_RGBA8;
        format = GL_RGBA;
        type = GL_UNSIGNED_BYTE;
        break;

    case FMT_RGB_F16:
        internal_format = GL_RGB16F;
        format = GL_RGB;
        type = GL_FLOAT;
        break;
    case FMT_RGBA_F16:
        internal_format = GL_RGBA16F;
        format = GL_RGBA;
        type = GL_FLOAT;
        break;

    case FMT_RGB_F32:
        internal_format = GL_RGB32F;
        format = GL_RGB;
        type = GL_FLOAT;
        break;
    case FMT_RGBA_F32:
        internal_format = GL_RGBA32F;
        format = GL_RGBA;
        type = GL_FLOAT;
        break;

    case FMT_DEPTH_F32:
        internal_format = GL_DEPTH_COMPONENT;
        format = GL_DEPTH_COMPONENT;
        type = GL_FLOAT;
        break;
    default:
        IST_ASSERT("unknown format: %d", fmt);
        return false;
    }

    m_width = width;
    m_height = height;
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

void Texture2D::bind(SLOT n) const
{
    glActiveTexture(GL_TEXTURE0+n);
    glBindTexture(GL_TEXTURE_2D, m_handle);
}
void Texture2D::unbind(SLOT n) const
{
    glActiveTexture(GL_TEXTURE0+n);
    glBindTexture(GL_TEXTURE_2D, m_handle);
}

GLuint Texture2D::getHandle() const { return m_handle; }
GLsizei Texture2D::getWidth() const { return m_width; }
GLsizei Texture2D::getHeight() const { return m_height; }



RenderBuffer::RenderBuffer()
: m_handle(0)
, m_width(0)
, m_height(0)
{
}

RenderBuffer::~RenderBuffer()
{
    finalize();
}

bool RenderBuffer::initialize()
{
    glGenRenderbuffers(1, &m_handle);
    return true;
}

bool RenderBuffer::initialize(GLsizei width, GLsizei height, FORMAT fmt)
{
    return initialize() && allocate(width, height, fmt);
}

void RenderBuffer::finalize()
{
    if(m_handle!=0) {
        glDeleteRenderbuffers(1, &m_handle);
    }
    m_handle = 0;
}

bool RenderBuffer::allocate(GLsizei width, GLsizei height, FORMAT fmt)
{
    GLint internal_format = 0;
    switch(fmt)
    {
    case FMT_RGB_U8:
    case FMT_RGB_F16:
    case FMT_RGB_F32:
        internal_format = GL_RGB;
        break;
    case FMT_RGBA_U8:
    case FMT_RGBA_F16:
    case FMT_RGBA_F32:
        internal_format = GL_RGBA;
        break;
    case FMT_DEPTH_F32:
        internal_format = GL_DEPTH_COMPONENT32;
        break;
    default:
        IST_ASSERT("unknown format: %d", fmt);
        return false;
    }

    m_width = width;
    m_height = height;
    glBindRenderbuffer( GL_RENDERBUFFER, m_handle );
    glRenderbufferStorage( GL_RENDERBUFFER, internal_format, width, height );
    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
    return true;
}

void RenderBuffer::bind() const
{
    glBindRenderbuffer(GL_RENDERBUFFER, m_handle);
}

void RenderBuffer::unbind() const
{
    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
}

GLuint RenderBuffer::getHandle() const { return m_handle; }
GLsizei RenderBuffer::getWidth() const { return m_width; }
GLsizei RenderBuffer::getHeight() const { return m_height; }



FrameBufferObject::FrameBufferObject()
: m_handle(0)
{
    std::fill_n(m_attaches, _countof(m_attaches), 0);
}

FrameBufferObject::~FrameBufferObject()
{
    finalize();
}

void FrameBufferObject::initialize()
{
    glGenFramebuffers(1, &m_handle);
}

void FrameBufferObject::finalize()
{
    if(m_handle!=0) {
        glDeleteFramebuffers(1, &m_handle);
    }
    m_handle = 0;
}

bool FrameBufferObject::attachRenderBuffer(RenderBuffer& rb, ATTACH attach)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attach, GL_RENDERBUFFER, rb.getHandle());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if(attach!=ATTACH_DEPTH) {
        for(int i=0; i<_countof(m_attaches); ++i) {
            if(m_attaches[i]==attach) { break; }
            if(m_attaches[i]==0) { m_attaches[i]=attach; break; }
        }
    }
    return true;
}

bool FrameBufferObject::attachTexture(Texture2D& tex, ATTACH attach, GLint level)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attach, GL_TEXTURE_2D, tex.getHandle(), level);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if(attach!=ATTACH_DEPTH) {
        for(int i=0; i<_countof(m_attaches); ++i) {
            if(m_attaches[i]==attach) { break; }
            if(m_attaches[i]==0) { m_attaches[i]=attach; break; }
        }
    }
    return true;
}

void FrameBufferObject::bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);

    int i=0;
    for(; i<_countof(m_attaches); ++i) {
        if(m_attaches[i]==0) { break; }
    }
    glDrawBuffers(i, m_attaches);
}

void FrameBufferObject::unbind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace graphics
} // namespace ist


