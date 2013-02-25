#include "istPCH.h"
#include "EasyDrawer.h"
#include "EasyDrawerShaders.h"

ist_EasyDrawer_NamespaceBegin


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
    struct VertexStreamData
    {
        uint32 stride;
        uint32 num_descs;
        VertexDesc descs[8];
    };


    static EasyShaders* getInstance()
    {
        if(s_inst==NULL) {
            s_inst = istNew(EasyShaders)();
        }
        s_inst->addRef();
        return s_inst;
    }

    ShaderProgram* getShader(VertexType vt)             { return m_shaders[vt]; }
    ShaderProgram* getShaderNonTextured(VertexType vt)  { return m_shaders[vt]; }
    VertexStreamData& getVertexInfo(VertexType vt)      { return m_vsdata[vt]; }

private:
    EasyShaders()
    {
        Device *dev = GetDevice();
        istMemset(m_shaders, 0, sizeof(m_shaders));
        istMemset(m_shaders_nt, 0, sizeof(m_shaders_nt));

        VertexShader *vs_p2c4   = CreateVertexShaderFromString(dev, g_vs_p2c4);
        VertexShader *vs_p2t2c4 = CreateVertexShaderFromString(dev, g_vs_p2t2c4);
        VertexShader *vs_p3t2c4 = CreateVertexShaderFromString(dev, g_vs_p3t2c4);
        PixelShader  *ps_c      = CreatePixelShaderFromString(dev, g_ps_colored);
        PixelShader  *ps_ct     = CreatePixelShaderFromString(dev, g_ps_colored_textured);

        m_shaders[VT_P2C4]      = dev->createShaderProgram(ShaderProgramDesc(vs_p2c4, ps_c));
        m_shaders[VT_P2T2C4]    = dev->createShaderProgram(ShaderProgramDesc(vs_p2t2c4, ps_ct));
        m_shaders[VT_P3T2C4]    = dev->createShaderProgram(ShaderProgramDesc(vs_p3t2c4, ps_ct));

        m_shaders_nt[VT_P2C4]   = m_shaders[VT_P2C4]; m_shaders[VT_P2C4]->addRef();
        m_shaders_nt[VT_P2T2C4] = dev->createShaderProgram(ShaderProgramDesc(vs_p2t2c4, ps_c));
        m_shaders_nt[VT_P3T2C4] = dev->createShaderProgram(ShaderProgramDesc(vs_p3t2c4, ps_c));

        istSafeRelease(ps_ct);
        istSafeRelease(ps_c);
        istSafeRelease(vs_p3t2c4);
        istSafeRelease(vs_p2t2c4);
        istSafeRelease(vs_p2c4);


        istMemset(m_vsdata, 0, sizeof(m_vsdata));
        {
            typedef VertexP2C4 Vertex;
            const VertexDesc descs[] = {
                {0, I3D_FLOAT32, 2,  0, false, 0},
                {1, I3D_FLOAT32, 4,  8, false, 0},
            };
            VertexStreamData &vsd = m_vsdata[GetVertexType<Vertex>::result];
            vsd.stride = sizeof(Vertex);
            vsd.num_descs = _countof(descs);
            std::copy(descs, descs+vsd.num_descs, vsd.descs);
        }
        {
            typedef VertexP2T2C4 Vertex;
            const VertexDesc descs[] = {
                {0, I3D_FLOAT32, 2,  0, false, 0},
                {1, I3D_FLOAT32, 2,  8, false, 0},
                {2, I3D_FLOAT32, 4, 16, false, 0},
            };
            VertexStreamData &vsd = m_vsdata[GetVertexType<Vertex>::result];
            vsd.stride = sizeof(Vertex);
            vsd.num_descs = _countof(descs);
            std::copy(descs, descs+vsd.num_descs, vsd.descs);
        }
        {
            typedef VertexP3T2C4 Vertex;
            const VertexDesc descs[] = {
                {0, I3D_FLOAT32, 3,  0, false, 0},
                {1, I3D_FLOAT32, 2, 12, false, 0},
                {2, I3D_FLOAT32, 4, 20, false, 0},
            };
            VertexStreamData &vsd = m_vsdata[GetVertexType<Vertex>::result];
            vsd.stride = sizeof(Vertex);
            vsd.num_descs = _countof(descs);
            std::copy(descs, descs+vsd.num_descs, vsd.descs);
        }
    }

    ~EasyShaders()
    {
        s_inst = NULL;
        for(size_t i=0; i<_countof(m_shaders); ++i) {
            istSafeRelease(m_shaders[i]);
            istSafeRelease(m_shaders_nt[i]);
        }
    }

    static EasyShaders *s_inst;
    ShaderProgram *m_shaders[VT_End];
    ShaderProgram *m_shaders_nt[VT_End];
    VertexStreamData m_vsdata[VT_End];
};
EasyShaders* EasyShaders::s_inst;


struct EasyDrawer::DrawCall
{
    EasyDrawState state;
    I3D_TOPOLOGY topology;
    VertexType vertex_type;
    uint32 num_vertices;
    size_t vb_offset; // in byte
    uint32 num_indices;
    size_t ib_offset; // in byte
};

struct EasyDrawer::Members
{
    typedef EasyDrawer::DrawCall DrawCall;

    VertexArray     *va;
    Buffer          *vbo;
    Buffer          *ibo;
    Buffer          *ubo;
    EasyShaders     *shaders;
    raw_vector<char>        vertex_data;
    raw_vector<char>        index_data;
    raw_vector<DrawCall>    draw_calls;

    Members()
        : va(NULL)
        , vbo(NULL)
        , ibo(NULL)
        , ubo(NULL)
        , shaders(NULL)
    {
    }
};

EasyDrawer* CreateEasyDrawer()
{
    return istNew(EasyDrawer)();
}

EasyDrawer::EasyDrawer()
{
    Device *dev = GetDevice();
    istSafeAddRef(dev);

    m->ubo = CreateUniformBuffer(dev, 256, I3D_USAGE_DYNAMIC);
    m->va = dev->createVertexArray();
    m->shaders = EasyShaders::getInstance();
}

EasyDrawer::~EasyDrawer()
{
    istSafeRelease(m->shaders);
    istSafeRelease(m->va);
    istSafeRelease(m->ubo);
    istSafeRelease(m->ibo);
    istSafeRelease(m->vbo);

    Device *dev = GetDevice();
    istSafeRelease(dev);
}

void EasyDrawer::release()
{
    istDelete(this);
}

template<class VertexT>
void EasyDrawer::draw( const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices )
{
    draw<VertexT>(states, topology, vertices, num_vertices, NULL, 0);
}

template<class VertexT>
void EasyDrawer::draw( const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint32 *indices, uint32 num_indices )
{
    if(num_vertices==0) { return; }

    VertexType vertex_type = GetVertexType<VertexT>::result;
    DrawCall tmp_cmd = {
        states,
        topology,
        vertex_type,
        num_vertices,
        m->vertex_data.size(),
        num_indices,
        m->index_data.size(),
    };
    if(tmp_cmd.state.getShader()==NULL) {
        if(tmp_cmd.state.getTexture()) {
            tmp_cmd.state.setShader(m->shaders->getShader(vertex_type));
        }
        else {
            tmp_cmd.state.setShader(m->shaders->getShaderNonTextured(vertex_type));
        }
    }
    m->draw_calls.push_back(tmp_cmd);
    {
        const char *v = (const char*)vertices;
        m->vertex_data.insert(m->vertex_data.end(), v, v+(sizeof(VertexT)*num_vertices));
    }
    if(num_indices>0) {
        const char *v = (const char*)indices;
        m->index_data.insert(m->index_data.end(), v, v+(sizeof(uint32)*num_indices));
    }
}

template void EasyDrawer::draw<VertexP2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2C4 *vertices, uint32 num_vertices);
template void EasyDrawer::draw<VertexP2T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2T2C4 *vertices, uint32 num_vertices);
template void EasyDrawer::draw<VertexP3T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP3T2C4 *vertices, uint32 num_vertices);

template void EasyDrawer::draw<VertexP2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2C4 *vertices, uint32 num_vertices, const uint32 *indices, uint32 num_indices);
template void EasyDrawer::draw<VertexP2T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP2T2C4 *vertices, uint32 num_vertices, const uint32 *indices, uint32 num_indices);
template void EasyDrawer::draw<VertexP3T2C4>(const EasyDrawState &states, I3D_TOPOLOGY topology, const VertexP3T2C4 *vertices, uint32 num_vertices, const uint32 *indices, uint32 num_indices);


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

        EasyShaders::VertexStreamData &vsd = m->shaders->getVertexInfo(dc.vertex_type);
        m->va->setAttributes(0, m->vbo, dc.vb_offset, vsd.stride, vsd.descs, vsd.num_descs);

        if(dc.num_indices==0) {
            ctx->draw(dc.topology, 0, dc.num_vertices);
        }
        else {
            ctx->setIndexBuffer(m->ibo, dc.ib_offset, I3D_UINT32);
            ctx->drawIndexed(dc.topology, dc.num_indices);
            ctx->setIndexBuffer(NULL, 0, I3D_UINT32);
        }
    }

    m->draw_calls.clear();
    m->vertex_data.clear();
    m->index_data.clear();
}

void EasyDrawer::updateBuffers(DeviceContext *ctx)
{
    Device *dev = GetDevice();
    if(!m->vertex_data.empty()) {
        uint32 vb_size = std::max<uint32>((uint32)m->vertex_data.size(), 1024*8);
        if(!m->vbo || m->vbo->getDesc().size<vb_size) {
            istSafeRelease(m->vbo);
            m->vbo = CreateVertexBuffer(dev, vb_size*2, I3D_USAGE_DYNAMIC);
        }
        MapAndWrite(ctx, m->vbo, &m->vertex_data[0], m->vertex_data.size());
    }

    if(!m->index_data.empty()) {
        uint32 ib_size = std::max<uint32>((uint32)m->index_data.size(), 1024*8);
        if(!m->ibo || m->ibo->getDesc().size<ib_size) {
            istSafeRelease(m->ibo);
            m->ibo = CreateIndexBuffer(dev, ib_size*2, I3D_USAGE_DYNAMIC);
        }
        MapAndWrite(ctx, m->ibo, &m->index_data[0], m->index_data.size());
    }
}


ist_EasyDraw_NamespaceEnd
