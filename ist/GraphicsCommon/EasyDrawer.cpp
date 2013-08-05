#include "istPCH.h"
#include "EasyDrawer.h"
#include "EasyDrawerShaders.h"

ist_EasyDrawer_NamespaceBegin


EasyDrawState::EasyDrawState()
    : m_texture(nullptr)
    , m_sampler(nullptr)
    , m_shader(nullptr)
    , m_uniform_location(0)
{
}

void EasyDrawState::setViewport( int32 width, int32 height )
{
    m_viewport = Viewport(ivec2(0,0), uvec2(width,height));
}
void EasyDrawState::setViewport( int32 x, int32 y, int32 width, int32 height )
{
    m_viewport = Viewport(ivec2(x,y), uvec2(width,height));
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

const Viewport& EasyDrawState::getViewport() const          { return m_viewport; }
const mat4&     EasyDrawState::getProjectionMatrix() const  { return m_proj; }
const mat4&     EasyDrawState::getWorldMatrix() const       { return m_world; }
Texture2D*      EasyDrawState::getTexture() const           { return m_texture; }
Sampler*        EasyDrawState::getSampler() const           { return m_sampler; }
ShaderProgram*  EasyDrawState::getShader() const            { return m_shader; }
uint32          EasyDrawState::getUniformLocation() const   { return m_uniform_location; }


class EasyDrawer::EasyShaders : public SharedObject
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
        if(s_inst==nullptr) {
            s_inst = istNew(EasyShaders)();
        }
        s_inst->addRef();
        return s_inst;
    }

    ShaderProgram* getShader(VertexType vt)             { return m_shaders[vt]; }
    ShaderProgram* getShaderNonTextured(VertexType vt)  { return m_shaders_nt[vt]; }
    VertexStreamData& getVertexInfo(VertexType vt)      { return m_vsdata[vt]; }

private:
    EasyShaders()
    {
        setRef(0);

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
            VertexStreamData &vsd = m_vsdata[GetVertexTypeID<Vertex>::result];
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
            VertexStreamData &vsd = m_vsdata[GetVertexTypeID<Vertex>::result];
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
            VertexStreamData &vsd = m_vsdata[GetVertexTypeID<Vertex>::result];
            vsd.stride = sizeof(Vertex);
            vsd.num_descs = _countof(descs);
            std::copy(descs, descs+vsd.num_descs, vsd.descs);
        }
    }

    ~EasyShaders()
    {
        s_inst = nullptr;
        for(size_t i=0; i<_countof(m_shaders); ++i) {
            istSafeRelease(m_shaders[i]);
            istSafeRelease(m_shaders_nt[i]);
        }
    }

    static EasyShaders *s_inst;
    ShaderProgram      *m_shaders[VT_End];
    ShaderProgram      *m_shaders_nt[VT_End];
    VertexStreamData    m_vsdata[VT_End];
};
EasyDrawer::EasyShaders* EasyDrawer::EasyShaders::s_inst;



const EasyDrawState& EasyDrawer::getRenderStates() {return m_state; }
void EasyDrawer::forceSetRenderStates( const EasyDrawState &ds ) { m_state=ds; }

void EasyDrawer::setViewport( int32 width, int32 height )               { m_state.setViewport(width,height); }
void EasyDrawer::setViewport( int32 x, int32 y, int32 w, int32 h )      { m_state.setViewport(x,y, w,h);}
void EasyDrawer::setScreen(float32 width, float32 height)               { m_state.setScreen(width, height); }
void EasyDrawer::setScreen(float32 l, float32 r, float32 b, float32 t)  { m_state.setScreen(l,r, b,t); }
void EasyDrawer::setProjectionMatrix(const mat4 &mat)                   { m_state.setProjectionMatrix(mat); }
void EasyDrawer::setWorldMatrix(const mat4 &mat)                        { m_state.setWorldMatrix(mat); }
void EasyDrawer::setTexture(Texture2D *tex)                             { m_state.setTexture(tex); }
void EasyDrawer::setSampler(Sampler *smp)                               { m_state.setSampler(smp); }
void EasyDrawer::setShader(ShaderProgram *smp)                          { m_state.setShader(smp); }

const Viewport& EasyDrawer::getViewport() const         { return m_state.getViewport(); }
const mat4&     EasyDrawer::getProjectionMatrix() const { return m_state.getProjectionMatrix(); }
const mat4&     EasyDrawer::getWorldMatrix() const      { return m_state.getWorldMatrix(); }
Texture2D*      EasyDrawer::getTexture() const          { return m_state.getTexture(); }
Sampler*        EasyDrawer::getSampler() const          { return m_state.getSampler(); }
ShaderProgram*  EasyDrawer::getShader() const           { return m_state.getShader(); }
uint32          EasyDrawer::getUniformLocation() const  { return m_state.getUniformLocation(); }


EasyDrawer* CreateEasyDrawer()
{
    return istNew(EasyDrawer)();
}

EasyDrawer::EasyDrawer()
    : m_va(nullptr)
    , m_vbo(nullptr)
    , m_ibo(nullptr)
    , m_ubo(nullptr)
    , m_shaders(nullptr)
{
    Device *dev = GetDevice();
    istSafeAddRef(dev);

    m_ubo = CreateUniformBuffer(dev, 256, I3D_USAGE_DYNAMIC);
    m_va = dev->createVertexArray();
    m_shaders = EasyShaders::getInstance();
}

EasyDrawer::~EasyDrawer()
{
    istSafeRelease(m_shaders);
    istSafeRelease(m_va);
    istSafeRelease(m_ubo);
    istSafeRelease(m_ibo);
    istSafeRelease(m_vbo);

    Device *dev = GetDevice();
    istSafeRelease(dev);
}

template<class T> struct GetIndexType {};
template<> struct GetIndexType<uint8>  { static const I3D_TYPE result = I3D_UINT8; };
template<> struct GetIndexType<uint16> { static const I3D_TYPE result = I3D_UINT16; };
template<> struct GetIndexType<uint32> { static const I3D_TYPE result = I3D_UINT32; };

template<class VertexT>
void EasyDrawer::drawImpl( I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices )
{
    drawImpl<VertexT, uint8>(topology, vertices, num_vertices, nullptr, 0);
}

template<class VertexT, class IndexT>
void EasyDrawer::drawImpl( I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const IndexT *indices, uint32 num_indices )
{
    if(num_vertices==0) { return; }

    VertexType vertex_type = GetVertexTypeID<VertexT>::result;
    DrawCall tmp_cmd = {
        m_state,
        topology,
        vertex_type,
        GetIndexType<IndexT>::result,
        num_vertices,
        m_vertex_data.size(),
        num_indices,
        m_index_data.size(),
    };
    if(tmp_cmd.state.getShader()==nullptr) {
        if(tmp_cmd.state.getTexture()) {
            tmp_cmd.state.setShader(m_shaders->getShader(vertex_type));
        }
        else {
            tmp_cmd.state.setShader(m_shaders->getShaderNonTextured(vertex_type));
        }
    }
    m_draw_calls.push_back(tmp_cmd);
    {
        const char *v = (const char*)vertices;
        m_vertex_data.insert(m_vertex_data.end(), v, v+(sizeof(VertexT)*num_vertices));
    }
    if(num_indices>0) {
        const char *v = (const char*)indices;
        m_index_data.insert(m_index_data.end(), v, v+(sizeof(uint32)*num_indices));
    }
}

#define Template(VertexT)\
    void EasyDrawer::draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices) { drawImpl(topology, vertices, num_vertices); }\
    void EasyDrawer::draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint16 *indices, uint32 num_indices) { drawImpl(topology, vertices, num_vertices, indices, num_indices); }\
    void EasyDrawer::draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint32 *indices, uint32 num_indices) { drawImpl(topology, vertices, num_vertices, indices, num_indices); }
istEachVertexTypes(Template)
#undef Template


void EasyDrawer::flush(DeviceContext *ctx)
{
    updateBuffers(ctx);

    mat4 prev_viewproj;
    for(size_t i=0; i<m_draw_calls.size(); ++i) {
        DrawCall &dc = m_draw_calls[i];
        const Viewport &vp = dc.state.getViewport();
        if(vp.getSize().x>0 && vp.getSize().y>0) {
            ctx->setViewport(vp);
        }

        mat4 viewproj = dc.state.getProjectionMatrix() * dc.state.getWorldMatrix();
        if(viewproj!=prev_viewproj) {
            prev_viewproj = viewproj;
            if(ctx->getDevice()->getSpec()->needs_transpose) {
                mat4 tmp = glm::transpose(viewproj);
                MapAndWrite(ctx, m_ubo, &tmp, sizeof(tmp));
            }
            else {
                MapAndWrite(ctx, m_ubo, &viewproj, sizeof(viewproj));
            }
        }

        ShaderProgram *shader = dc.state.getShader();
        ctx->setShader(shader);
        ctx->setUniformBuffer(dc.state.getUniformLocation(), 0, m_ubo);
        ctx->setSampler(0, dc.state.getSampler());
        ctx->setTexture(0, dc.state.getTexture());
        ctx->setVertexArray(m_va);

        EasyShaders::VertexStreamData &vsd = m_shaders->getVertexInfo(dc.vertex_type);
        m_va->setAttributes(0, m_vbo, dc.vb_offset, vsd.stride, vsd.descs, vsd.num_descs);

        if(dc.num_indices==0) {
            ctx->draw(dc.topology, 0, dc.num_vertices);
        }
        else {
            ctx->setIndexBuffer(m_ibo, dc.ib_offset, dc.index_type);
            ctx->drawIndexed(dc.topology, dc.num_indices);
            ctx->setIndexBuffer(nullptr, 0, I3D_UINT32);
        }
    }

    m_draw_calls.clear();
    m_vertex_data.clear();
    m_index_data.clear();
}

void EasyDrawer::updateBuffers(DeviceContext *ctx)
{
    Device *dev = GetDevice();
    if(!m_vertex_data.empty()) {
        uint32 vb_size = std::max<uint32>((uint32)m_vertex_data.size(), 1024*8);
        if(!m_vbo || m_vbo->getDesc().size<vb_size) {
            istSafeRelease(m_vbo);
            m_vbo = CreateVertexBuffer(dev, vb_size*2, I3D_USAGE_DYNAMIC);
        }
        MapAndWrite(ctx, m_vbo, &m_vertex_data[0], m_vertex_data.size());
    }

    if(!m_index_data.empty()) {
        uint32 ib_size = std::max<uint32>((uint32)m_index_data.size(), 1024*8);
        if(!m_ibo || m_ibo->getDesc().size<ib_size) {
            istSafeRelease(m_ibo);
            m_ibo = CreateIndexBuffer(dev, ib_size*2, I3D_USAGE_DYNAMIC);
        }
        MapAndWrite(ctx, m_ibo, &m_index_data[0], m_index_data.size());
    }
}


ist_EasyDraw_NamespaceEnd
