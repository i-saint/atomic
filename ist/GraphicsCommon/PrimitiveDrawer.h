#ifndef ist_Graphics_PrimitiveDrawer_h
#define ist_Graphics_PrimitiveDrawer_h

#include "ist/Base.h"
#include "ist/Graphics.h"

namespace ist {

#ifdef ist_with_OpenGL
    namespace i3d = ::ist::i3dgl;
#endif // ist_with_OpenGL



struct VertexP2T2C4
{
    vec2 position;
    vec2 texcoord;
    vec4 color;
};

enum VERTEX_TYPE {
    VERTEX_UNKNOWN,
    VERTEX_P2T2C4,
};

struct PrimitiveDrawStates
{
    float left, right, top, bottom;
};


class istInterModule IPrimitiveDrawer
{
public:
    virtual ~IPrimitiveDrawer();
    virtual void draw(i3d::I3D_TOPOLOGY topology, const VertexP2T2C4 *vertices, uint32 num_vertices)=0;
    virtual void flush()=0;
};

class PrimitiveDrawer : public IPrimitiveDrawer
{
public:
    PrimitiveDrawer();
    virtual ~PrimitiveDrawer();
    virtual void draw(const PrimitiveDrawStates &states, i3d::I3D_TOPOLOGY topology, const VertexP2T2C4 *vertices, uint32 num_vertices);
    virtual void flush();

private:
    struct DrawCommand
    {
        PrimitiveDrawStates states;
        I3D_TOPOLOGY topology;
        uint32 vertex_start;
        uint32 vertex_num;
    };

    stl::vector<VertexP2T2C4> m_vertices_p2t2c4;
    i3d::Buffer *m_vertex_buffer;
    i3d::ShaderProgram *m_shader;
};


} // namespace ist

#endif // ist_Graphics_PrimitiveDrawer_h
