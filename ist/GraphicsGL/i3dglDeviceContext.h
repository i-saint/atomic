#ifndef __ist_i3dgl_DeviceContext__
#define __ist_i3dgl_DeviceContext__

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"
#include "i3dglBuffer.h"
#include "i3dglRenderTarget.h"
#include "i3dglShader.h"

namespace ist {
namespace i3dgl {

class DeviceContext
{
I3DGL_DECLARE_DEVICE_RESOURCE(DeviceContext);
private:
    Device         *m_device;

    Buffer         *m_ia_vertex_buffers[I3D_MAX_VERTEX_BUFFERS];
    uint32          m_ia_vertex_strides[I3D_MAX_VERTEX_BUFFERS];
    Buffer         *m_ia_index_buffer;
    I3D_TYPE        m_ia_index_type;
    VertexArray    *m_ia_vertex_array;
    I3D_TOPOLOGY    m_ia_topology;

    VertexShader    *m_vs_shader;

    PixelShader     *m_ps_shader;

    RenderTarget    *m_om_rendertargets;

private:
    DeviceContext(Device *dev);
    ~DeviceContext();

public:

    void setRenderTarget(RenderBuffer **rt, uint32 num, RenderBuffer *depthstencil);

    void IAsetVertexBuffer(Buffer **vb, uint32 *strides, uint32 num);
    void IAsetInputLayout();
    void IAsetIndexBuffer(Buffer *v, I3D_TYPE format);

    void VSsetVertexShader(VertexShader *v);

    void PSsetPixelShader(PixelShader *v);
    void PSsetSamplers();
    void PSsetTextures(uint32 location, Texture2D *textures, uint32 num_textures);
    void PSsetConstantBuffers();

    void draw(uint32 num_vertices);
    void drawIndexed(uint32 num_indices);
    void drawInstanced(uint32 num_vertices, uint32 num_instances);
    void drawIndexedInstanced(uint32 num_indices, uint32 num_instances);
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_DeviceContext__
