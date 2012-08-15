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
public:
    void setRenderTarget(uint32 num, RenderBuffer **rt, RenderBuffer *depthstencil);

    void IAsetVertexBuffers(uint32 num_vbo, Buffer **vb, uint32 *strides);
    void IAsetInputLayout(uint32 num_descs, const VertexDesc *descs);
    void IAsetIndexBuffer(Buffer *v, I3D_TYPE format);
    void IAsetPrimitiveTopology(I3D_TOPOLOGY v);

    void VSsetShader(VertexShader *v);

    void PSsetShader(PixelShader *v);
    void PSsetSamplers();
    void PSsetTextures(uint32 location, Texture2D *textures, uint32 num_textures);
    void PSsetConstantBuffers();

    void draw(uint32 num_vertices, uint32 first_vertex);
    void drawIndexed(uint32 num_indices, uint32 first_vertex);
    void drawInstanced(uint32 num_vertices, uint32 first_vertex, uint32 num_instances);
    void drawIndexedInstanced(uint32 num_indices, uint32 first_vertex, uint32 num_instances);

private:
    DeviceContext(Device *dev);
    ~DeviceContext();

    void prepareDraw();

private:
    Device         *m_device;

    VertexArray     *m_va;

    uint32          m_ia_num_vertex_buffers;
    Buffer         *m_ia_vertex_buffers[I3D_MAX_VERTEX_BUFFERS];
    uint32          m_ia_vertex_strides[I3D_MAX_VERTEX_BUFFERS];
    uint32          m_ia_num_vertex_descs;
    VertexDesc      m_ia_vertex_descs[I3D_MAX_VERTEX_DESCS];
    Buffer         *m_ia_index_buffer;
    I3D_TYPE        m_ia_index_format;
    VertexArray    *m_ia_vertex_array;
    uint32          m_ia_num_render_targets;
    RenderTarget   *m_ia_render_target;
    RenderBuffer   *m_ia_frame_buffers[I3D_MAX_RENDER_TARGETS];
    RenderBuffer   *m_ia_depth_stencil_buffer;

    I3D_TOPOLOGY    m_ia_topology;
    ShaderProgram   *m_shader;
    VertexShader    *m_vs_shader;
    PixelShader     *m_ps_shader;
    RenderTarget    *m_om_rendertargets;
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_DeviceContext__
