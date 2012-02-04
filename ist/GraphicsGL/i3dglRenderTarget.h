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
