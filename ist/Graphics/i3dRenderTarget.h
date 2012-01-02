#ifndef __ist_i3d_RenderTarget__
#define __ist_i3d_RenderTarget__

#include "i3dTypes.h"
#include "i3dDeviceResource.h"

namespace ist {
namespace i3d {


class Texture2D : public DeviceResource
{
I3D_DECLARE_DEVICE_RESOURCE(Texture2D)
private:
    GLsizei m_width;
    GLsizei m_height;

private:
    Texture2D();
    ~Texture2D();

public:
    bool allocate(GLsizei width, GLsizei height, I3D_COLOR_FORMAT format, void *data=NULL);

    void bind() const;
    void unbind() const;
    void bind(int slot) const;  // slot: preferred to IST_TEXTURE_SLOT
    void unbind(int slot) const;// slot: preferred to IST_TEXTURE_SLOT

    GLsizei getWidth() const;
    GLsizei getHeight() const;
};



class RenderBuffer : public DeviceResource
{
I3D_DECLARE_DEVICE_RESOURCE(RenderBuffer)
private:
    GLsizei m_width;
    GLsizei m_height;

    RenderBuffer();
    ~RenderBuffer();

public:
    bool allocate(GLsizei width, GLsizei height, I3D_COLOR_FORMAT format);

    void bind() const;
    void unbind() const;

    GLsizei getWidth() const;
    GLsizei getHeight() const;
};


class RenderTarget : public DeviceResource
{
I3D_DECLARE_DEVICE_RESOURCE(RenderTarget)
public:
    static const int32 MAX_RENDER_BUFFERS = 16;

private:
    Texture2D *m_color_buffers[MAX_RENDER_BUFFERS];
    Texture2D *m_depthstencil;
    uint32 m_num_color_buffers;

    RenderTarget();
    ~RenderTarget();

    void releaseBuffers();

public:
    bool setRenderBuffers(Texture2D **rb, uint32 num, Texture2D *depthstencil);
    bool getRenderBuffers(Texture2D **rb, uint32 &num, Texture2D *&depthstencil);
    void setNumColorBuffers(uint32 v);
    void setColorBuffer(uint32 i, Texture2D *rb);
    void setDepthStencilBuffer(Texture2D *rb);
    uint32 getNumColorBuffers() const;
    Texture2D* getColorBuffer(uint32 i);
    Texture2D* getDepthStencilBuffer();
    void bind() const;
    void unbind() const;

    // RenderBuffer ‚ÍŽg‚í‚È‚¢ & DirectX ‚É‘Š“–‹@”\‚È‚¢‚Ì‚Å”ñ‘Î‰ž‚Å
    //bool attachRenderBuffer(RenderBuffer& tex, I3D_RT_ATTACH attach);
};


} // namespace graphics
} // namespace ist
#endif // __ist_i3d_RenderTarget__
