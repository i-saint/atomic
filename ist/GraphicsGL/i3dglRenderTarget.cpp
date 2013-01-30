#include "istPCH.h"
#include "ist/Base.h"
#ifdef ist_with_OpenGL
#include "i3dglTexture.h"
#include "i3dglRenderTarget.h"

namespace ist {
namespace i3dgl {

bool DetectGLFormat(I3D_COLOR_FORMAT fmt, GLint &internal_format, GLint &format, GLint &type)
{
    switch(fmt)
    {
    case I3D_R8:        internal_format=GL_R8;          format=GL_RED;  type=GL_UNSIGNED_BYTE;  break;
    case I3D_R8S:       internal_format=GL_R8_SNORM;    format=GL_RED;  type=GL_BYTE;           break;
    case I3D_R8U:       internal_format=GL_R8UI;        format=GL_RED;  type=GL_UNSIGNED_BYTE;  break;
    case I3D_R8I:       internal_format=GL_R8I;         format=GL_RED;  type=GL_BYTE;           break;
    case I3D_R16F:      internal_format=GL_R16F;        format=GL_RED;  type=GL_FLOAT;          break;
    case I3D_R32F:      internal_format=GL_R32F;        format=GL_RED;  type=GL_FLOAT;          break;

    case I3D_RG8:       internal_format=GL_RG8;         format=GL_RG;   type=GL_UNSIGNED_BYTE;  break;
    case I3D_RG8S:      internal_format=GL_RG8_SNORM;   format=GL_RG;   type=GL_BYTE;           break;
    case I3D_RG8U:      internal_format=GL_RG8UI;       format=GL_RG;   type=GL_UNSIGNED_BYTE;  break;
    case I3D_RG8I:      internal_format=GL_RG8I;        format=GL_RG;   type=GL_BYTE;           break;
    case I3D_RG16F:     internal_format=GL_RG16F;       format=GL_RG;   type=GL_FLOAT;          break;
    case I3D_RG32F:     internal_format=GL_RG32F;       format=GL_RG;   type=GL_FLOAT;          break;

    case I3D_RGB8:      internal_format=GL_RGB8;        format=GL_RGB;  type=GL_UNSIGNED_BYTE;  break;
    case I3D_RGB8S:     internal_format=GL_RGB8_SNORM;  format=GL_RGB;  type=GL_BYTE;           break;
    case I3D_RGB8U:     internal_format=GL_RGB8UI;      format=GL_RGB;  type=GL_UNSIGNED_BYTE;  break;
    case I3D_RGB8I:     internal_format=GL_RGB8I;       format=GL_RGB;  type=GL_BYTE;           break;
    case I3D_RGB16F:    internal_format=GL_RGB16F;      format=GL_RGB;  type=GL_FLOAT;          break;
    case I3D_RGB32F:    internal_format=GL_RGB32F;      format=GL_RGB;  type=GL_FLOAT;          break;

    case I3D_RGBA8:     internal_format=GL_RGBA8;       format=GL_RGBA; type=GL_UNSIGNED_BYTE;  break;
    case I3D_RGBA8S:    internal_format=GL_RGBA8_SNORM; format=GL_RGBA; type=GL_BYTE;           break;
    case I3D_RGBA8U:    internal_format=GL_RGBA8UI;     format=GL_RGBA; type=GL_UNSIGNED_BYTE;  break;
    case I3D_RGBA8I:    internal_format=GL_RGBA8I;      format=GL_RGBA; type=GL_BYTE;           break;
    case I3D_RGBA16F:   internal_format=GL_RGBA16F;     format=GL_RGBA; type=GL_FLOAT;          break;
    case I3D_RGBA32F:   internal_format=GL_RGBA32F;     format=GL_RGBA; type=GL_FLOAT;          break;

    case I3D_DEPTH16F:          internal_format=GL_DEPTH_COMPONENT16;   format=GL_DEPTH_COMPONENT;  type=GL_FLOAT;                          break;
    case I3D_DEPTH32F:          internal_format=GL_DEPTH_COMPONENT;     format=GL_DEPTH_COMPONENT;  type=GL_FLOAT;                          break;
    case I3D_DEPTH24_STENCIL8:  internal_format=GL_DEPTH24_STENCIL8;    format=GL_DEPTH_STENCIL;    type=GL_UNSIGNED_INT_24_8;              break;
    case I3D_DEPTH32F_STENCIL8: internal_format=GL_DEPTH32F_STENCIL8;   format=GL_DEPTH_STENCIL;    type=GL_FLOAT_32_UNSIGNED_INT_24_8_REV; break;

    case I3D_RGB_DXT1:  internal_format=GL_COMPRESSED_RGB_S3TC_DXT1_EXT;        break;
    case I3D_SRGB_DXT1: internal_format=GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;       break;
    case I3D_RGBA_DXT1: internal_format=GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;       break;
    case I3D_SRGBA_DXT1:internal_format=GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT; break;
    case I3D_RGBA_DXT3: internal_format=GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;       break;
    case I3D_SRGBA_DXT3:internal_format=GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT; break;
    case I3D_RGBA_DXT5: internal_format=GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;       break;
    case I3D_SRGBA_DXT5:internal_format=GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT; break;
    default:
        istAssert(false);
        return false;
    }
    return true;
}



RenderTarget::RenderTarget(Device *dev)
    : super(dev)
    , m_num_color_buffers(0)
{
    glGenFramebuffers(1, &m_handle);
    stl::fill_n(m_color_buffers, _countof(m_color_buffers), (Texture2D*)NULL);
    stl::fill_n(m_color_mips, _countof(m_color_mips), 0);
    m_depthstencil = NULL;
    m_depthstencil_mips = 0;
    m_dirty.flags = 0;
}

RenderTarget::~RenderTarget()
{
    releaseBuffers();
    if(m_handle!=0) {
        glDeleteFramebuffers(1, &m_handle);
        m_handle = 0;
    }
}

void RenderTarget::releaseBuffers()
{
    istSafeRelease(m_depthstencil);
    for(uint32 i=0; i<m_num_color_buffers; ++i) {
        istSafeRelease(m_color_buffers[i]);
    }
}

bool RenderTarget::setRenderBuffers(Texture2D **color, uint32 num_color, Texture2D *depthstencil, uint32 level)
{
    if(num_color>=I3D_MAX_RENDER_TARGETS) {
        istPrint("number of render targets must be less than %d\n", I3D_MAX_RENDER_TARGETS);
        return false;
    }

    // 現バッファと rb, depthstencil が同じ物を指してる可能性があるため、先に参照カウンタ増加
    for(uint32 i=0; i<num_color; ++i){ istSafeAddRef(color[i]); }
    istSafeAddRef(depthstencil);

    // 現バッファの参照カウンタ減少
    releaseBuffers();

    // copy & dirty flags
    stl::copy(color, color+num_color, m_color_buffers);
    stl::fill(m_color_mips, m_color_mips+num_color, level);
    m_num_color_buffers = num_color;
    m_depthstencil = depthstencil;
    m_depthstencil_mips = level;

    m_dirty.color = 1;
    m_dirty.num_color = 1;
    m_dirty.depthstencil = 1;

    return true;
}

bool RenderTarget::getRenderBuffers(Texture2D **rb, uint32 &num, Texture2D *&depthstencil)
{
    if(num < m_num_color_buffers) { return false; }

    num = m_num_color_buffers;
    stl::copy(m_color_buffers, m_color_buffers+m_num_color_buffers, rb);
    depthstencil = m_depthstencil;
    return true;
}

void RenderTarget::setNumColorBuffers(uint32 v)
{
    m_num_color_buffers = v;

    m_dirty.num_color = 1;
}

void RenderTarget::setColorBuffer(uint32 slot, Texture2D *color, uint32 level)
{
    istAssert(slot<_countof(m_color_buffers));

    istSafeAddRef(color);
    istSafeRelease(m_color_buffers[slot]);

    m_color_buffers[slot] = color;
    m_color_mips[slot] = level;

    uint32 used = 0;
    for(uint32 slot=0; slot<_countof(m_color_buffers); ++slot) {
        if(m_color_buffers[slot]) { used = slot+1; }
    }
    m_num_color_buffers = used;

    m_dirty.color = 1;
}

void RenderTarget::setDepthStencilBuffer(Texture2D *depthstencil, uint32 level)
{
    istSafeAddRef(depthstencil);
    istSafeRelease(m_depthstencil);

    m_depthstencil = depthstencil;
    m_depthstencil_mips = level;

    m_dirty.depthstencil = 1;
}

void RenderTarget::setMipmapLevel( int32 level )
{
    stl::fill_n(m_color_mips, _countof(m_color_mips), level);
    m_depthstencil_mips = level;

    m_dirty.color = 1;
    m_dirty.depthstencil = 1;
}


uint32 RenderTarget::getNumColorBuffers() const
{
    return m_num_color_buffers;
}

Texture2D* RenderTarget::getColorBuffer(uint32 i)
{
    if(i >= m_num_color_buffers) { return NULL; }
    return m_color_buffers[i];
}

Texture2D* RenderTarget::getDepthStencilBuffer()
{
    return m_depthstencil;
}

void RenderTarget::bind()
{
    m_dirty.flags = 0;

    static const GLuint attaches[16] = {
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2,
        GL_COLOR_ATTACHMENT3,
        GL_COLOR_ATTACHMENT4,
        GL_COLOR_ATTACHMENT5,
        GL_COLOR_ATTACHMENT6,
        GL_COLOR_ATTACHMENT7,
        GL_COLOR_ATTACHMENT8,
        GL_COLOR_ATTACHMENT9,
        GL_COLOR_ATTACHMENT10,
        GL_COLOR_ATTACHMENT11,
        GL_COLOR_ATTACHMENT12,
        GL_COLOR_ATTACHMENT13,
        GL_COLOR_ATTACHMENT14,
        GL_COLOR_ATTACHMENT15,
    };

    glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
    for(uint32 i=0; i<m_num_color_buffers; ++i) {
        GLuint h = m_color_buffers[i] ? m_color_buffers[i]->getHandle() : 0;
        uint32 level = m_color_mips[i];
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, h, level);
    }
    {
        GLuint h = m_depthstencil ? m_depthstencil->getHandle() : 0;
        uint32 level = m_depthstencil_mips;
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, h, level);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, h, level);
    }
    glDrawBuffers(m_num_color_buffers, attaches);
}

void RenderTarget::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace i3d
} // namespace ist
#endif // ist_with_OpenGL
