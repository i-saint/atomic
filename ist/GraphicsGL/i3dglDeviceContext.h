#ifndef ist_i3dgl_DeviceContext_h
#define ist_i3dgl_DeviceContext_h

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
    struct RenderStates
    {
        RenderTarget        *render_target;
        VertexArray         *vertex_array;
        ShaderProgram       *shader;
        struct {
            Buffer          *buffer;
            size_t          offset;
            I3D_TYPE        format;
        } index;
        struct {
            Buffer          *buffer;
            int32           bind;
            bool            dirty;
        } uniform[8];
        Sampler             *samplers[8];
        Texture             *textures[16];
        BlendState          *blend_state;
        DepthStencilState   *depthstencil_state;
        Viewport            viewport;

        RenderStates() { istMemset(this, 0, sizeof(*this)); }
    };
    union DirtyFlags
    {
        struct {
            uint32 render_target:1;
            uint32 vertex_array:1;
            uint32 shader:1;
            uint32 index:1;
            uint32 uniform:1;
            uint32 samplers:1;
            uint32 textures:1;
            uint32 blend_state:1;
            uint32 depthstencil_state:1;
            uint32 viewport:1;
        };
        uint32 flags;
    };

public:
    void setViewport(const Viewport &vp);
    void setVertexArray(VertexArray *va);
    void setIndexBuffer(Buffer *v, size_t offset, I3D_TYPE format); // format: I3D_UBYTE/I3D_USHORT/I3D_UINT
    void setUniformBuffer(int32 loc, int32 bind, Buffer *buf);
    void setShader(ShaderProgram *v);
    void setRenderTarget(RenderTarget *rt);
    void setSampler(uint32 slot, Sampler *smp);
    void setTexture(uint32 slot, Texture *tex);
    void setBlendState(BlendState *state);
    void setDepthStencilState(DepthStencilState *state);

    void draw(I3D_TOPOLOGY topology, uint32 first_vertex, uint32 num_vertices);
    void drawIndexed(I3D_TOPOLOGY topology, uint32 num_indices);
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

    const RenderStates& getRenderStates() const;

private:
    DeviceContext(Device *dev);
    ~DeviceContext();

private:
    Device         *m_device;

    RenderStates m_prev;
    RenderStates m_current;
    DirtyFlags m_dirty;
};

} // namespace i3d
} // namespace ist

#endif // ist_i3dgl_DeviceContext_h
