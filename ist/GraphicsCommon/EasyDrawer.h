#ifndef ist_Graphics_EasyDrawer_h
#define ist_Graphics_EasyDrawer_h

#include "ist/Base.h"
#include "ist/Graphics.h"

#define ist_EasyDraw_impl_GL


namespace ist {
#ifdef ist_EasyDraw_impl_GL
namespace i3dgl {
#endif // ist_EasyDraw_impl_GL


struct VertexP3T2C4
{
    vec3 position;
    vec2 texcoord;
    vec4 color;
};

class EasyDrawStates
{
public:
    EasyDrawStates();
    void setScreen(float32 left, float32 right, float32 bottom, float32 top);

private:
    mat4 m_proj;
};


class istInterModule IEasyDrawer
{
public:
    virtual ~IEasyDrawer();
    virtual void release();
    virtual void draw(const EasyDrawStates &state, I3D_TOPOLOGY topology, const VertexP3T2C4 *vertices, uint32 num_vertices)=0;
    virtual void flush()=0;
};
IEasyDrawer* CreateEasyDrawer(Device *dev, DeviceContext *ctx);

} // namespace i3d*
} // namespace ist

#endif // ist_Graphics_EasyDrawer_h
