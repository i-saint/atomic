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
    RenderTarget    *m_rt;
    VertexArray     *m_va;
    VertexShader    *m_vs;
    PixelShader     *m_ps;
    GeometryShader  *m_gs;
    I3D_TOPOLOGY    m_topology;

public:
    DeviceContext();

    void setRenderTarget(RenderTarget *rt);
    void setVertexArray(VertexArray *v);
    void setIndexBuffer(IndexBuffer *v);
    void setVertexShader(VertexShader *v);
    void setPixelShader(PixelShader *v);
    void setGeometryShader(GeometryShader *v);

    void draw(uint32 v_offset, uint32 v_num);
    void drawIndexed(uint32 v_offset, uint32 v_num);
    void drawInstanced(uint32 v_offset, uint32 v_num);
    void drawIndexedInstanced(uint32 v_offset, uint32 v_num, uint32 i_num);
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_DeviceContext__
