#ifndef __ist_i3ddx11_DeviceContext__
#define __ist_i3ddx11_DeviceContext__

#include "i3ddx11Types.h"
#include "i3ddx11Buffer.h"
#include "i3ddx11RenderTarget.h"
#include "i3ddx11Shader.h"

namespace ist {
namespace i3ddx11 {

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
    ~DeviceContext();

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

} // namespace i3ddx11
} // namespace ist

#endif // __ist_i3ddx11_DeviceContext__
