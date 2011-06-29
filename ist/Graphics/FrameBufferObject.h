#ifndef __ist_Graphics_FrameBufferObject__
#define __ist_Graphics_FrameBufferObject__

#include "GraphicsResource.h"

namespace ist {
namespace graphics {


class Texture2D : public GraphicsResource
{
public:
    enum FORMAT
    {
        FMT_RGB_U8,
        FMT_RGBA_U8,
        FMT_RGB_F16,
        FMT_RGBA_F16,
        FMT_RGB_F32,
        FMT_RGBA_F32,
        FMT_DEPTH_F32,
    };
    enum SLOT
    {
        SLOT_0,
        SLOT_1,
        SLOT_2,
        SLOT_3,
        SLOT_4,
        SLOT_5,
        SLOT_6,
        SLOT_7,
    };

private:
    GLuint m_handle;
    GLsizei m_width;
    GLsizei m_height;

public:
    Texture2D();
    ~Texture2D();

    bool initialize();
    bool initialize(GLsizei width, GLsizei height, FORMAT format, void *data=NULL);
    void finalize();
    bool allocate(GLsizei width, GLsizei height, FORMAT format, void *data=NULL);

    void bind() const;
    void unbind() const;
    void bind(SLOT n) const;
    void unbind(SLOT n) const;

    GLuint getHandle() const;
    GLsizei getWidth() const;
    GLsizei getHeight() const;
};



class RenderBuffer : public GraphicsResource
{
public:
    enum FORMAT
    {
        FMT_RGB_U8      = Texture2D::FMT_RGB_U8,
        FMT_RGBA_U8     = Texture2D::FMT_RGBA_U8,
        FMT_RGB_F16     = Texture2D::FMT_RGB_F16,
        FMT_RGBA_F16    = Texture2D::FMT_RGBA_F16,
        FMT_RGB_F32     = Texture2D::FMT_RGB_F32,
        FMT_RGBA_F32    = Texture2D::FMT_RGBA_F32,
        FMT_DEPTH_F32   = Texture2D::FMT_DEPTH_F32,
    };
private:
    GLuint m_handle;
    GLsizei m_width;
    GLsizei m_height;

public:
    RenderBuffer();
    ~RenderBuffer();

    bool initialize();
    bool initialize(GLsizei width, GLsizei height, FORMAT format);
    void finalize();
    bool allocate(GLsizei width, GLsizei height, FORMAT format);

    void bind() const;
    void unbind() const;

    GLuint getHandle() const;
    GLsizei getWidth() const;
    GLsizei getHeight() const;
};


class FrameBufferObject : public GraphicsResource
{
public:
    enum ATTACH
    {
        ATTACH_COLOR0 = GL_COLOR_ATTACHMENT0,
        ATTACH_COLOR1 = GL_COLOR_ATTACHMENT1,
        ATTACH_COLOR2 = GL_COLOR_ATTACHMENT2,
        ATTACH_COLOR3 = GL_COLOR_ATTACHMENT3,
        ATTACH_COLOR4 = GL_COLOR_ATTACHMENT4,
        ATTACH_COLOR5 = GL_COLOR_ATTACHMENT5,
        ATTACH_COLOR6 = GL_COLOR_ATTACHMENT6,
        ATTACH_COLOR7 = GL_COLOR_ATTACHMENT7,
        ATTACH_COLOR8 = GL_COLOR_ATTACHMENT8,
        ATTACH_COLOR9 = GL_COLOR_ATTACHMENT9,
        ATTACH_COLOR10 = GL_COLOR_ATTACHMENT10,
        ATTACH_COLOR11 = GL_COLOR_ATTACHMENT11,
        ATTACH_COLOR12 = GL_COLOR_ATTACHMENT12,
        ATTACH_COLOR13 = GL_COLOR_ATTACHMENT13,
        ATTACH_COLOR14 = GL_COLOR_ATTACHMENT14,
        ATTACH_COLOR15 = GL_COLOR_ATTACHMENT15,
        ATTACH_DEPTH  = GL_DEPTH_ATTACHMENT,
    };
private:
    GLuint m_handle;
    GLuint m_attaches[18]; // 

public:
    FrameBufferObject();
    ~FrameBufferObject();

    void initialize();
    void finalize();

    bool attachRenderBuffer(RenderBuffer& tex, ATTACH attach);
    bool attachTexture(Texture2D& rb, ATTACH attach, GLint level=0);
    void bind() const;
    void unbind() const;

    GLuint getHandle() const { return m_handle; }
};


} // graphics
} // ist
#endif // __ist_Graphics_FrameBufferObject__
