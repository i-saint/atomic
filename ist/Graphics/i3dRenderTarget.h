#ifndef __ist_i3d_RenderTarget__
#define __ist_i3d_RenderTarget__

#include "i3dTypes.h"
#include "i3dDeviceResource.h"

namespace ist {
namespace i3d {


class Texture2D : public DeviceResource
{
I3D_DECLARE_DEVICE_RESOURCE()
private:
    GLuint m_handle;
    GLsizei m_width;
    GLsizei m_height;

public:
    Texture2D();
    ~Texture2D();

    bool initialize();
    bool initialize(GLsizei width, GLsizei height, I3D_COLOR_FORMAT format, void *data=NULL);
    void finalize();
    bool allocate(GLsizei width, GLsizei height, I3D_COLOR_FORMAT format, void *data=NULL);

    void bind() const;
    void unbind() const;
    void bind(int slot) const;  // slot: preferred to IST_TEXTURE_SLOT
    void unbind(int slot) const;// slot: preferred to IST_TEXTURE_SLOT

    GLuint getHandle() const;
    GLsizei getWidth() const;
    GLsizei getHeight() const;
};



class RenderBuffer : public DeviceResource
{
private:
    GLuint m_handle;
    GLsizei m_width;
    GLsizei m_height;

public:
    RenderBuffer();
    ~RenderBuffer();

    bool initialize();
    bool initialize(GLsizei width, GLsizei height, I3D_COLOR_FORMAT format);
    void finalize();
    bool allocate(GLsizei width, GLsizei height, I3D_COLOR_FORMAT format);

    void bind() const;
    void unbind() const;

    GLuint getHandle() const;
    GLsizei getWidth() const;
    GLsizei getHeight() const;
};


class RenderTarget : public DeviceResource
{
private:
    GLuint m_handle;
    GLuint m_attaches; // 0bit-15bit –Ú‚ª‚»‚ê‚¼‚ê ATTACH_COLOR0-ATTACH_COLOR15 ‚É‘Î‰ž

public:
    RenderTarget();
    ~RenderTarget();

    bool initialize();
    void finalize();

    bool attachRenderBuffer(RenderBuffer& tex, I3D_RT_ATTACH attach);
    bool attachTexture(Texture2D& rb, I3D_RT_ATTACH attach, GLint level=0);
    void bind() const;
    void unbind() const;

    GLuint getHandle() const { return m_handle; }
};


} // namespace graphics
} // namespace ist
#endif // __ist_i3d_RenderTarget__
