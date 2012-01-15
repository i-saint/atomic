#ifndef __ist_i3dgl_DeviceContext__
#define __ist_i3dgl_DeviceContext__

#include "i3dglTypes.h"
#include "i3dglBuffer.h"
#include "i3dglRenderTarget.h"
#include "i3dglShader.h"

namespace ist {
namespace i3dgl {

class DeviceContext
{
private:
    Buffer          *m_vertex_buffers[I3D_MAX_VERTEX_BUFFERS];
    Buffer          *m_index_buffer;
    VertexArray     *m_vertex_array;
    I3D_TOPOLOGY    m_topology;

    VertexShader    *m_vertex_shader;
    PixelShader     *m_pixel_shader;

    RenderTarget    *m_render_target;

public:
    DeviceContext();

    void setRenderTarget(Texture2D **rt, uint32 num, Texture2D *depthstencil);

    void IAsetVertexBuffer(Buffer **vb, uint32 *strides, uint32 num);
    void IAsetInputLayout();
    void IAsetIndexBuffer(Buffer *v, I3D_TYPE format);

    void setVertexShader(VertexShader *v);
    void setPixelShader(PixelShader *v);

    void draw(uint32 v_offset, uint32 v_num);
    void drawIndexed(uint32 v_offset, uint32 v_num);
    void drawInstanced(uint32 v_offset, uint32 v_num);
    void drawIndexedInstanced(uint32 v_offset, uint32 v_num, uint32 i_num);
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_DeviceContext__
