#ifndef __ist_Graphics_FrameBufferObject__
#define __ist_Graphics_FrameBufferObject__

#include "GraphicsResource.h"

namespace ist {
namespace graphics {

enum IST_COLOR_FORMAT
{
    IST_UNKNOWN,
    IST_R8U,
    IST_R16F,
    IST_R32F,
    IST_RG8U,
    IST_RG16F,
    IST_RG32F,
    IST_RGB8U,  // texture only
    IST_RGB16F, // 
    IST_RGB32F, // 
    IST_RGBA8U,
    IST_RGBA16F,
    IST_RGBA32F,
    IST_DEPTH32F,
    IST_DEPTH24_STENCIL8,
    IST_DEPTH32F_STENCIL8,
};

enum IST_RT_ATTACH
{
    IST_ATTACH_COLOR0   = GL_COLOR_ATTACHMENT0,
    IST_ATTACH_COLOR1   = GL_COLOR_ATTACHMENT1,
    IST_ATTACH_COLOR2   = GL_COLOR_ATTACHMENT2,
    IST_ATTACH_COLOR3   = GL_COLOR_ATTACHMENT3,
    IST_ATTACH_COLOR4   = GL_COLOR_ATTACHMENT4,
    IST_ATTACH_COLOR5   = GL_COLOR_ATTACHMENT5,
    IST_ATTACH_COLOR6   = GL_COLOR_ATTACHMENT6,
    IST_ATTACH_COLOR7   = GL_COLOR_ATTACHMENT7,
    IST_ATTACH_COLOR8   = GL_COLOR_ATTACHMENT8,
    IST_ATTACH_COLOR9   = GL_COLOR_ATTACHMENT9,
    IST_ATTACH_COLOR10  = GL_COLOR_ATTACHMENT10,
    IST_ATTACH_COLOR11  = GL_COLOR_ATTACHMENT11,
    IST_ATTACH_COLOR12  = GL_COLOR_ATTACHMENT12,
    IST_ATTACH_COLOR13  = GL_COLOR_ATTACHMENT13,
    IST_ATTACH_COLOR14  = GL_COLOR_ATTACHMENT14,
    IST_ATTACH_COLOR15  = GL_COLOR_ATTACHMENT15,
    IST_ATTACH_DEPTH    = GL_DEPTH_ATTACHMENT,
    IST_ATTACH_STENCIL  = GL_STENCIL_ATTACHMENT,
};

enum IST_TEXTURE_SLOT
{
    IST_TEX_SLOT_0,
    IST_TEX_SLOT_1,
    IST_TEX_SLOT_2,
    IST_TEX_SLOT_3,
    IST_TEX_SLOT_4,
    IST_TEX_SLOT_5,
    IST_TEX_SLOT_6,
    IST_TEX_SLOT_7,
};


class Texture2D : public GraphicsResource
{
private:
    GLuint m_handle;
    GLsizei m_width;
    GLsizei m_height;

public:
    Texture2D();
    ~Texture2D();

    bool initialize();
    bool initialize(GLsizei width, GLsizei height, IST_COLOR_FORMAT format, void *data=NULL);
    void finalize();
    bool allocate(GLsizei width, GLsizei height, IST_COLOR_FORMAT format, void *data=NULL);

    void bind() const;
    void unbind() const;
    void bind(int slot) const;  // slot: preferred to IST_TEXTURE_SLOT
    void unbind(int slot) const;// slot: preferred to IST_TEXTURE_SLOT

    GLuint getHandle() const;
    GLsizei getWidth() const;
    GLsizei getHeight() const;
};



class RenderBuffer : public GraphicsResource
{
private:
    GLuint m_handle;
    GLsizei m_width;
    GLsizei m_height;

public:
    RenderBuffer();
    ~RenderBuffer();

    bool initialize();
    bool initialize(GLsizei width, GLsizei height, IST_COLOR_FORMAT format);
    void finalize();
    bool allocate(GLsizei width, GLsizei height, IST_COLOR_FORMAT format);

    void bind() const;
    void unbind() const;

    GLuint getHandle() const;
    GLsizei getWidth() const;
    GLsizei getHeight() const;
};


class FrameBufferObject : public GraphicsResource
{
private:
    GLuint m_handle;
    GLuint m_attaches; // 0bit-15bit –Ú‚ª‚»‚ê‚¼‚ê ATTACH_COLOR0-ATTACH_COLOR15 ‚É‘Î‰ž

public:
    FrameBufferObject();
    ~FrameBufferObject();

    bool initialize();
    void finalize();

    bool attachRenderBuffer(RenderBuffer& tex, IST_RT_ATTACH attach);
    bool attachTexture(Texture2D& rb, IST_RT_ATTACH attach, GLint level=0);
    void bind() const;
    void unbind() const;

    GLuint getHandle() const { return m_handle; }
};


} // graphics
} // ist
#endif // __ist_Graphics_FrameBufferObject__
