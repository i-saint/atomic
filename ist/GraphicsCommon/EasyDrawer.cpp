#include "istPCH.h"
#include "EasyDrawer.h"

namespace ist {
#ifdef ist_EasyDraw_impl_GL
namespace i3dgl {
#endif // ist_EasyDraw_impl_GL




#ifdef ist_EasyDraw_impl_GL
const char *g_vs_p2c4 = "\
#version 330 core\n\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
layout(location=0) in vec2 ia_VertexPosition;\
layout(location=1) in vec4 ia_VertexColor;\
out vec4 vs_Color;\
\
void main(void)\
{\
    vs_Color    = ia_VertexColor;\
    gl_Position = u_RS.ViewProjectionMatrix * vec4(ia_VertexPosition, 0.0f, 1.0);\
}\
";

const char *g_vs_p2t2c4 = "\
#version 330 core\n\
struct RenderStates\
{\
    mat4 ViewProjectionMatrix;\
};\
layout(std140) uniform render_states\
{\
    RenderStates u_RS;\
};\
layout(location=0) in vec2 ia_VertexPosition;\
layout(location=1) in vec2 ia_VertexTexcoord;\
layout(location=2) in vec4 ia_VertexColor;\
out vec2 vs_Texcoord;\
out vec4 vs_Color;\
\
void main(void)\
{\
    vs_Texcoord = ia_VertexTexcoord;\
    vs_Color    = ia_VertexColor;\
    gl_Position = u_RS.ViewProjectionMatrix * vec4(ia_VertexPosition, 0.0f, 1.0);\
}\
";

const char *g_ps_colored = "\
#version 330 core\n\
uniform sampler2D u_Texture;\
in vec4 vs_Color;\
layout(location=0) out vec4 ps_FragColor;\
\
void main()\
{\
    vec4 color = vs_Color;\
    ps_FragColor = vec4(color);\
}\
";

const char *g_ps_colored_textured = "\
#version 330 core\n\
uniform sampler2D u_Texture;\
in vec2 vs_Texcoord;\
in vec4 vs_Color;\
layout(location=0) out vec4 ps_FragColor;\
\
void main()\
{\
    vec4 color = vs_Color;\
    color *= texture(u_Texture, vs_Texcoord);\
    ps_FragColor = vec4(color);\
}\
";
#endif // ist_EasyDraw_impl_GL


EasyDrawState::EasyDrawState()
    : m_texture(NULL)
    , m_sampler(NULL)
    , m_shader(NULL)
    , m_uniform_location(0)
{
}

void EasyDrawState::setScreen(float32 width, float32 height)
{
    setScreen(0.0f, width, height, 0.0f);
}
void EasyDrawState::setScreen(float32 left, float32 right, float32 bottom, float32 top)
{
    m_proj = glm::ortho(left, right, bottom, top);
}
void EasyDrawState::setProjectionMatrix(const mat4 &mat){ m_proj=mat; }
void EasyDrawState::setWorldMatrix(const mat4 &mat)     { m_world=mat; }
void EasyDrawState::setTexture(Texture2D *tex)          { m_texture=tex; }
void EasyDrawState::setSampler(Sampler *smp)            { m_sampler=smp; }
void EasyDrawState::setShader(ShaderProgram *v)
{
    m_shader = v;
    if(m_shader) {
        m_uniform_location = m_shader->getUniformBlockIndex("render_states");
    }
}

const mat4&     EasyDrawState::getProjectionMatrix() const  { return m_proj; }
const mat4&     EasyDrawState::getWorldMatrix() const       { return m_world; }
Texture2D*      EasyDrawState::getTexture() const           { return m_texture; }
Sampler*        EasyDrawState::getSampler() const           { return m_sampler; }
ShaderProgram*  EasyDrawState::getShader() const            { return m_shader; }
uint32          EasyDrawState::getUniformLocation() const   { return m_uniform_location; }


class EasyShaders : public SharedObject
{
public:
    static EasyShaders* getInstance(Device *dev)
    {
        if(s_inst==NULL) {
            s_inst = istNew(EasyShaders)(dev);
        }
        s_inst->addRef();
        return s_inst;
    }

    ShaderProgram* getShader(VertexType vt)
    {
        return m_shaders[vt];
    }

private:
    EasyShaders(Device *dev)
    {
        istMemset(m_shaders, 0, sizeof(m_shaders));
        {
            VertexShader *vs = CreateVertexShaderFromString(dev, g_vs_p2c4);
            PixelShader  *ps = CreatePixelShaderFromString(dev, g_ps_colored);
            m_shaders[VT_P2C4] = dev->createShaderProgram(ShaderProgramDesc(vs, ps));
            istSafeRelease(ps);
            istSafeRelease(vs);
        }
        {
            VertexShader *vs = CreateVertexShaderFromString(dev, g_vs_p2t2c4);
            PixelShader  *ps = CreatePixelShaderFromString(dev, g_ps_colored_textured);
            m_shaders[VT_P2T2C4] = dev->createShaderProgram(ShaderProgramDesc(vs, ps));
            istSafeRelease(ps);
            istSafeRelease(vs);
        }
    }

    ~EasyShaders()
    {
        s_inst = NULL;
        for(size_t i=0; i<_countof(m_shaders); ++i) {
            if(m_shaders[i]) {
                m_shaders[i]->release();
            }
        }
    }

    static EasyShaders *s_inst;
    ShaderProgram *m_shaders[VT_End];
};
EasyShaders* EasyShaders::s_inst;

struct EasyDrawer::Members
{
    typedef EasyDrawer::DrawCall DrawCall;

    Device          *dev;
    VertexArray     *va;
    Buffer          *vbo;
    Buffer          *ubo;
    EasyShaders     *shaders;
    raw_vector<char>        vertex_data;
    raw_vector<DrawCall>    draw_calls;

    Members()
        : dev(NULL)
        , va(NULL)
        , vbo(NULL)
        , ubo(NULL)
        , shaders(NULL)
    {
    }
};

EasyDrawer* CreateEasyDrawer(Device *dev)
{
    return istNew(EasyDrawer)(dev);
}

EasyDrawer::EasyDrawer(Device *dev)
{
    m->dev = dev;
    istSafeAddRef(m->dev);
    m->ubo = CreateUniformBuffer(m->dev, 256, I3D_USAGE_DYNAMIC);
    m->va = m->dev->createVertexArray();
    m->shaders = EasyShaders::getInstance(dev);
}

EasyDrawer::~EasyDrawer()
{
    istSafeRelease(m->shaders);
    istSafeRelease(m->va);
    istSafeRelease(m->ubo);
    istSafeRelease(m->vbo);
    istSafeRelease(m->dev);
}

void EasyDrawer::release()
{
    istDelete(this);
}

template<class VertexT>
void EasyDrawer::draw( const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices )
{
    VertexType vertex_type = GetVertexType<VertexT>::result;
    DrawCall tmp_cmd = {
        states,
        topology,
        vertex_type,
        num_vertices,
        m->vertex_data.size(),
    };
    if(tmp_cmd.state.getShader()==NULL) {
        tmp_cmd.state.setShader(m->shaders->getShader(vertex_type));
    }
    m->draw_calls.push_back(tmp_cmd);

    const char *v = (const char*)vertices;
    m->vertex_data.insert(m->vertex_data.end(), v, v+(sizeof(VertexT)*num_vertices));
}
template void EasyDrawer::draw<VertexP2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2C4 *vertices, uint32 num_vertices);
template void EasyDrawer::draw<VertexP2T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2T2C4 *vertices, uint32 num_vertices);
template void EasyDrawer::draw<VertexP3T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP3T2C4 *vertices, uint32 num_vertices);


void EasyDrawer::flush(DeviceContext *ctx)
{
    updateBuffers(ctx);

    mat4 prev_viewproj;
    for(size_t i=0; i<m->draw_calls.size(); ++i) {
        DrawCall &dc = m->draw_calls[i];

        mat4 viewproj = dc.state.getProjectionMatrix() * dc.state.getWorldMatrix();
        if(viewproj!=prev_viewproj) {
            prev_viewproj = viewproj;
            MapAndWrite(ctx, m->ubo, &viewproj, sizeof(viewproj));
        }

        ShaderProgram *shader = dc.state.getShader();
        ctx->setShader(shader);
        ctx->setUniformBuffer(dc.state.getUniformLocation(), 0, m->ubo);
        ctx->setSampler(0, dc.state.getSampler());
        ctx->setTexture(0, dc.state.getTexture());
        ctx->setVertexArray(m->va);

        switch(dc.vertex_type) {
        case VT_P2C4:
            {
                const VertexDesc descs[] = {
                    {0, I3D_FLOAT, 2,  0, false, 0},
                    {1, I3D_FLOAT, 4,  8, false, 0},
                };
                m->va->setAttributes(0, m->vbo, dc.buffer_index, sizeof(VertexP2C4), descs, _countof(descs));
            }
            break;
        case VT_P2T2C4:
            {
                const VertexDesc descs[] = {
                    {0, I3D_FLOAT, 2,  0, false, 0},
                    {1, I3D_FLOAT, 2,  8, false, 0},
                    {2, I3D_FLOAT, 4, 16, false, 0},
                };
                m->va->setAttributes(0, m->vbo, dc.buffer_index, sizeof(VertexP2T2C4), descs, _countof(descs));
            }
            break;
        case VT_P3T2C4:
            {
                const VertexDesc descs[] = {
                    {0, I3D_FLOAT, 3,  0, false, 0},
                    {1, I3D_FLOAT, 2, 12, false, 0},
                    {2, I3D_FLOAT, 4, 20, false, 0},
                };
                m->va->setAttributes(0, m->vbo, dc.buffer_index, sizeof(VertexP3T2C4), descs, _countof(descs));
            }
            break;
        }

        ctx->draw(dc.topology, 0, dc.num_vertices);
    }

    m->draw_calls.clear();
    m->vertex_data.clear();
}

void EasyDrawer::updateBuffers(DeviceContext *ctx)
{
    uint32 vb_size = std::max<uint32>((uint32)m->vertex_data.size(), 1024*8);
    if(!m->vbo || m->vbo->getDesc().size<vb_size) {
        istSafeRelease(m->vbo);
        m->vbo = CreateVertexBuffer(m->dev, vb_size*2, I3D_USAGE_DYNAMIC);
    }
    MapAndWrite(ctx, m->vbo, &m->vertex_data[0], m->vertex_data.size());
}

} // namespace i3d*
} // namespace ist
