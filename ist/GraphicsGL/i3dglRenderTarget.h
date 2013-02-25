#ifndef ist_i3dgl_RenderTarget_h
#define ist_i3dgl_RenderTarget_h

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {


class RenderTarget : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(RenderTarget)
typedef DeviceResource super;
public:
    bool setRenderBuffers(Texture2D **color, uint32 num_color, Texture2D *depthstencil, uint32 level=0);
    bool getRenderBuffers(Texture2D **color, uint32 &num_color, Texture2D *&depthstencil);
    void setNumColorBuffers(uint32 v);
    void setColorBuffer(uint32 slot, Texture2D *color, uint32 level=0);
    void setDepthStencilBuffer(Texture2D *depthstencil, uint32 level=0);
    void setMipmapLevel(int32 level);
    uint32 getNumColorBuffers() const;
    Texture2D* getColorBuffer(uint32 slot);
    Texture2D* getDepthStencilBuffer();


    // RenderBuffer は使わない & DirectX に相当機能ないので非対応で
    //bool attachRenderBuffer(RenderBuffer& tex, I3D_RT_ATTACH attach);

private:
    Texture2D *m_color_buffers[I3D_MAX_RENDER_TARGETS];
    uint32 m_color_mips[I3D_MAX_RENDER_TARGETS];
    uint32 m_num_color_buffers;
    Texture2D *m_depthstencil;
    mutable uint32 m_depthstencil_mips;
    union {
        struct {
            uint32 color:1;
            uint32 num_color:1;
            uint32 depthstencil:1;
        };
        uint32 flags;
    } m_dirty;

    RenderTarget(Device *dev);
    ~RenderTarget();

    void releaseBuffers();
    void bind();
    void unbind();
    bool isDirty() const { return m_dirty.flags!=0; }
};


} // namespace graphics
} // namespace ist
#endif // ist_i3dgl_RenderTarget_h
