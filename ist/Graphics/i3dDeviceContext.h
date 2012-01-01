#ifndef __ist_i3d_DeviceContext__
#define __ist_i3d_DeviceContext__

#include "i3dTypes.h"
#include "i3dBuffer.h"
#include "i3dRenderTarget.h"
#include "i3dShader.h"

namespace ist {
namespace i3d {

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

#endif // __ist_i3d_DeviceContext__
