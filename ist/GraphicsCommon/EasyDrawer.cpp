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

    void updateBuffers();

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
    Buffer *m_vbo;
    Buffer *m_ubo;
    ShaderProgram *m_shader;
    GLint m_uniform_loc;
};

IEasyDrawer* CreateEasyDrawer(Device *dev, DeviceContext *ctx)
{
    return istNew(EasyDrawer)(dev, ctx);
}

EasyDrawer::EasyDrawer(Device *dev, DeviceContext *ctx)
    : m_dev(dev)
    , m_ctx(ctx)
    , m_vbo(NULL)
    , m_ubo(NULL)
    , m_shader(NULL)
    , m_uniform_loc(0)
{
    istSafeAddRef(m_dev);
    istSafeAddRef(m_ctx);
    m_ubo = CreateUniformBuffer(m_dev, 256, I3D_USAGE_DYNAMIC);

    VertexShader *vs = CreateVertexShaderFromString(m_dev, g_vs);
    PixelShader *ps = CreatePixelShaderFromString(m_dev, g_ps);
    m_shader = m_dev->createShaderProgram(ShaderProgramDesc(vs, ps));
    istSafeRelease(vs);
    istSafeRelease(ps);
    m_uniform_loc = m_shader->getUniformBlockIndex("render_states");
}

EasyDrawer::~EasyDrawer()
{
    istSafeRelease(m_shader);
    istSafeRelease(m_ubo);
    istSafeRelease(m_vbo);
    istSafeRelease(m_ctx);
    istSafeRelease(m_dev);
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
    updateBuffers();
    m_ctx->setShader(m_shader);
    //m_ctx->setVertexArray();
    for(size_t i=0; i<m_commands.size(); ++i) {
        DrawCommand &cmd = m_commands[i];
        m_ctx->draw(cmd.topology, cmd.vertex_start/sizeof(VertexP3T2C4), cmd.vertex_num);
    }

    m_commands.clear();
    m_raw_vertices.clear();
}

void EasyDrawer::updateBuffers()
{
    uint32 vb_size = std::max<uint32>((uint32)m_raw_vertices.size(), 1024*8);
    if(!m_vbo || m_vbo->getDesc().size<vb_size) {
        istSafeRelease(m_vbo);
        m_vbo = CreateVertexBuffer(m_dev, vb_size*2, I3D_USAGE_DYNAMIC);
    }
    MapAndWrite(m_ctx, m_vbo, &m_raw_vertices, m_raw_vertices.size());
}

} // namespace i3d*
} // namespace ist
