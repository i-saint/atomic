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
    void setViewport(const Viewport &vp);
    void setVertexArray(VertexArray *va);
    void setIndexBuffer(Buffer *v, I3D_TYPE format);
    void setShader(ShaderProgram *v);
    void setRenderTarget(RenderTarget *rt);
    void setSampler(uint32 i, Sampler *smp);
    void setTexture(uint32 i, Texture *tex);

    void draw(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices);
    void drawIndexed(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_indices);
    void drawInstanced(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices, uint32 num_instances);
    void drawIndexedInstanced(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_indices, uint32 num_instances);
    void applyRenderStates();

    void clearColor(RenderTarget *rt, vec4 color);
    void clearDepth(RenderTarget *rt, float32 depth);
    void clearStencil(RenderTarget *rt, int32 stencil);

private:
    DeviceContext(Device *dev);
    ~DeviceContext();


private:
    Device         *m_device;

    RenderTarget   *m_render_target;
    VertexArray    *m_vertex_array;
    ShaderProgram  *m_shader;

    Buffer         *m_index_buffer;
    I3D_TYPE        m_index_format;
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_DeviceContext__
