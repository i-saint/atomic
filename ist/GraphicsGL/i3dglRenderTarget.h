#ifndef __ist_i3dgl_RenderTarget__
#define __ist_i3dgl_RenderTarget__

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {


class Sampler : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Sampler)
typedef DeviceResource super;
private:
public:
    Sampler();
    ~Sampler();
    void bind(uint32 slot);
    void unbind(uint32 slot);
};


class Texture1D : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Texture1D)
typedef DeviceResource super;
private:
    uint32 m_size;

private:
    Texture1D(Device *dev);
    ~Texture1D();

public:
    bool allocate(uint32 size, I3D_COLOR_FORMAT format, void *data=NULL);
    void copy(uint32 mip_level, uint32 pos, uint32 size, I3D_COLOR_FORMAT format, void *data);

    void bind() const;
    void unbind() const;
    void bind(int slot) const;
    void unbind(int slot) const;

    uint32 getSize() const;
};

class Texture2D : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Texture2D)
typedef DeviceResource super;
private:
    uvec2 m_size;

private:
    Texture2D(Device *dev);
    ~Texture2D();

public:
    bool allocate(const uvec2 &size, I3D_COLOR_FORMAT format, void *data=NULL);
    void copy(uint32 mip_level, const uvec2 &pos, const uvec2 &size, I3D_COLOR_FORMAT format, void *data);

    void bind() const;
    void unbind() const;
    void bind(int slot) const;
    void unbind(int slot) const;

    const uvec2& getSize() const;
};



class RenderTarget : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(RenderTarget)
typedef DeviceResource super;
private:
    Texture2D *m_color_buffers[I3D_MAX_RENDER_TARGETS];
    Texture2D *m_depthstencil;
    uint32 m_num_color_buffers;

    RenderTarget(Device *dev);
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
#endif // __ist_i3dgl_RenderTarget__
