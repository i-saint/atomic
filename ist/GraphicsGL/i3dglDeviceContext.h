#ifndef __ist_i3dgl_DeviceContext__
#define __ist_i3dgl_DeviceContext__

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"
#include "i3dglBuffer.h"
#include "i3dglRenderTarget.h"
#include "i3dglShader.h"

namespace ist {
namespace i3dgl {

class DeviceContext : public SharedObject
{
I3DGL_DECLARE_DEVICE_RESOURCE(DeviceContext);
public:
    void setViewport(const Viewport &vp);
    void setVertexArray(VertexArray *va);
    void setIndexBuffer(Buffer *v, I3D_TYPE format);
    void setShader(ShaderProgram *v);
    void setRenderTarget(RenderTarget *rt);
    void setSampler(uint32 slot, Sampler *smp);
    void setTexture(uint32 slot, Texture *tex);
    void setBlendState(BlendState *state);
    void setDepthStencilState(DepthStencilState *state);

    void draw(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices);
    void drawIndexed(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_indices);
    void drawInstanced(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices, uint32 num_instances);
    void drawIndexedInstanced(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_indices, uint32 num_instances);
    void clearColor(RenderTarget *rt, vec4 color);
    void clearDepthStencil(RenderTarget *rt, float32 depth, int32 stencil);

    void* map(Buffer *buffer, I3D_MAP_MODE mode);
    void unmap(Buffer *buffer);
    void updateResource(Texture1D *tex, uint32 mip, uint32 pos, uint32 size, void *data); // size: 0 なら全領域。たぶんそのほうが速い
    void updateResource(Texture2D *tex, uint32 mip,  uvec2 pos,  uvec2 size, void *data); // 
    void updateResource(Texture3D *tex, uint32 mip,  uvec3 pos,  uvec3 size, void *data); // 
    void generateMips(Texture *tex);

    void applyRenderStates();

private:
    DeviceContext(Device *dev);
    ~DeviceContext();

private:
    Device         *m_device;

    struct RenderStates
    {
        RenderTarget        *render_target;
        VertexArray         *vertex_array;
        ShaderProgram       *shader;
        Buffer              *index_buffer;
        I3D_TYPE            index_format;
        BlendState          *blend_state;
        DepthStencilState   *depthstencil_state;
    };
    RenderStates m_prev;
    RenderStates m_current;

};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_DeviceContext__
