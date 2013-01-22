#include "istPCH.h"
#include "EasyDrawer.h"

namespace ist {
#ifdef ist_EasyDraw_impl_GL
namespace i3dgl {
#endif // ist_EasyDraw_impl_GL

#ifdef ist_EasyDraw_impl_GL
const char *g_vs = "\
#version 330 core\n\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
layout(location=0) in vec3 ia_VertexPosition;\
layout(location=1) in vec2 ia_VertexTexcoord;\
layout(location=2) in vec4 ia_VertexColor;\
out vec2 vs_Texcoord;\
out vec4 vs_Color;\
\
void main(void)\
{\
    vs_Texcoord = ia_VertexTexcoord;\
    vs_Color    = ia_VertexColor;\
    gl_Position = u_RS.ViewProjectionMatrix * vec4(ia_VertexPosition, 1.0);\
}\
";

const char *g_ps = "\
#version 330 core\n\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
uniform sampler2D u_Font;\
in vec2 vs_Texcoord;\
in vec4 vs_Color;\
layout(location=0) out vec4 ps_FragColor;\
\
void main()\
{\
    vec4 color = vs_Color;\
    color.a *= texture(u_Font, vs_Texcoord).r;\
    ps_FragColor = vec4(color);\
}\
";
#endif // ist_EasyDraw_impl_GL

class EasyDrawer : public IEasyDrawer
{
public:
    EasyDrawer(Device *dev, DeviceContext *ctx);
    virtual ~EasyDrawer();
    virtual void release();
    virtual void draw(const EasyDrawStates &states, I3D_TOPOLOGY topology, const VertexP3T2C4 *vertices, uint32 num_vertices);
    virtual void flush();

private:
    struct DrawCommand
    {
        EasyDrawStates states;
        I3D_TOPOLOGY topology;
        uint32 vertex_start;
        uint32 vertex_num;
    };

    stl::vector<char> m_raw_vertices;
    stl::vector<DrawCommand> m_commands;
    Device *m_dev;
    DeviceContext *m_ctx;
    Buffer *m_vertex_buffer;
    ShaderProgram *m_shader;
};

IEasyDrawer* CreateEasyDrawer(Device *dev, DeviceContext *ctx)
{
    return istNew(EasyDrawer)(dev, ctx);
}

EasyDrawer::EasyDrawer(Device *dev, DeviceContext *ctx)
    : m_dev(dev)
    , m_ctx(ctx)
    , m_vertex_buffer(NULL)
    , m_shader(NULL)
{
}

EasyDrawer::~EasyDrawer()
{
    istSafeRelease(m_shader);
    istSafeRelease(m_vertex_buffer);
}

void EasyDrawer::release()
{
    istDelete(this);
}

void EasyDrawer::draw( const EasyDrawStates &states, I3D_TOPOLOGY topology, const VertexP3T2C4 *vertices, uint32 num_vertices )
{
    DrawCommand tmp_cmd = {
        states,
        topology,
        static_cast<uint32>(m_raw_vertices.size()),
        num_vertices,
    };
    m_commands.push_back(tmp_cmd);

    const char *v = (const char*)vertices;
    m_raw_vertices.insert(m_raw_vertices.end(), v, v+(sizeof(VertexP3T2C4)*num_vertices));
}

void EasyDrawer::flush()
{

}

} // namespace i3d*
} // namespace ist
