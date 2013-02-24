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


struct EasyDrawer::Members
{
    typedef EasyDrawer::DrawCall DrawCall;

    Device          *dev;
    DeviceContext   *ctx;
    VertexArray     *va;
    Buffer          *vbo;
    Buffer          *ubo;
    ShaderProgram   *shader;
    GLint                   uniform_loc;
    raw_vector<char>        vertex_data;
    raw_vector<DrawCall>    draw_calls;

    Members()
        : dev(NULL)
        , ctx(NULL)
        , va(NULL)
        , vbo(NULL)
        , ubo(NULL)
        , shader(NULL)
        , uniform_loc(0)
    {
    }
};

EasyDrawer* CreateEasyDrawer(Device *dev, DeviceContext *ctx)
{
    return istNew(EasyDrawer)(dev, ctx);
}

EasyDrawer::EasyDrawer(Device *dev, DeviceContext *ctx)
{
    m->dev = dev;
    m->ctx = ctx;
    istSafeAddRef(m->dev);
    istSafeAddRef(m->ctx);
    m->ubo = CreateUniformBuffer(m->dev, 256, I3D_USAGE_DYNAMIC);
    m->va = m->dev->createVertexArray();

    VertexShader *vs = CreateVertexShaderFromString(m->dev, g_vs);
    PixelShader *ps = CreatePixelShaderFromString(m->dev, g_ps);
    m->shader = m->dev->createShaderProgram(ShaderProgramDesc(vs, ps));
    istSafeRelease(vs);
    istSafeRelease(ps);
    m->uniform_loc = m->shader->getUniformBlockIndex("render_states");
}

EasyDrawer::~EasyDrawer()
{
    istSafeRelease(m->shader);
    istSafeRelease(m->ubo);
    istSafeRelease(m->vbo);
    istSafeRelease(m->ctx);
    istSafeRelease(m->dev);
}

void EasyDrawer::release()
{
    istDelete(this);
}

template<class VertexT>
void EasyDrawer::draw( const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices )
{
    DrawCall tmp_cmd = {
        states,
        topology,
        GetVertexType<VertexT>::result,
        num_vertices,
        m->vertex_data.size(),
    };
    m->draw_calls.push_back(tmp_cmd);

    const char *v = (const char*)vertices;
    m->vertex_data.insert(m->vertex_data.end(), v, v+(sizeof(VertexT)*num_vertices));
}
template void EasyDrawer::draw<VertexP2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2C4 *vertices, uint32 num_vertices);
template void EasyDrawer::draw<VertexP2T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2T2C4 *vertices, uint32 num_vertices);
template void EasyDrawer::draw<VertexP3T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP3T2C4 *vertices, uint32 num_vertices);


void EasyDrawer::flush()
{
    updateBuffers();
    m->ctx->setShader(m->shader);
    m->ctx->setVertexArray(m->va);
    for(size_t i=0; i<m->draw_calls.size(); ++i) {
        DrawCall &dc = m->draw_calls[i];
        m->ctx->draw(dc.topology, 0, dc.num_vertices);
    }

    m->draw_calls.clear();
    m->vertex_data.clear();
}

void EasyDrawer::updateBuffers()
{
    uint32 vb_size = std::max<uint32>((uint32)m->vertex_data.size(), 1024*8);
    if(!m->vbo || m->vbo->getDesc().size<vb_size) {
        istSafeRelease(m->vbo);
        m->vbo = CreateVertexBuffer(m->dev, vb_size*2, I3D_USAGE_DYNAMIC);
    }
    MapAndWrite(m->ctx, m->vbo, &m->vertex_data, m->vertex_data.size());
}

} // namespace i3d*
} // namespace ist
