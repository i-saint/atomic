#ifndef __ist_i3dgl_RenderTarget__
#define __ist_i3dgl_RenderTarget__

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {


class RenderTarget : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(RenderTarget)
typedef DeviceResource super;
public:
    bool setRenderBuffers(Texture2D **rb, uint32 num, Texture2D *depthstencil, uint32 level=0);
    bool getRenderBuffers(Texture2D **rb, uint32 &num, Texture2D *&depthstencil);
    void setNumColorBuffers(uint32 v);
    void setColorBuffer(uint32 i, Texture2D *rb, uint32 level=0);
    void setDepthStencilBuffer(Texture2D *rb, uint32 level=0);
    void setMipmapLevel(int32 level);
    uint32 getNumColorBuffers() const;
    Texture2D* getColorBuffer(uint32 i);
    Texture2D* getDepthStencilBuffer();

    void bind() const;
    void unbind() const;

    // RenderBuffer は使わない & DirectX に相当機能ないので非対応で
    //bool attachRenderBuffer(RenderBuffer& tex, I3D_RT_ATTACH attach);

private:
    Texture2D *m_color_buffers[I3D_MAX_RENDER_TARGETS];
    Texture2D *m_depthstencil;
    uint32 m_num_color_buffers;

    RenderTarget(Device *dev);
    ~RenderTarget();

    void releaseBuffers();
};


} // namespace graphics
} // namespace ist
#endif // __ist_i3dgl_RenderTarget__
