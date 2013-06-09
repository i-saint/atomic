#include "istPCH.h"
#include "EasyDrawer.h"
#include "EasyDrawerShaders.h"

ist_EasyDrawer_NamespaceBegin


struct EasyDrawState::Members
{
    mat4            proj;
    mat4            world;
    Texture2D      *texture;
    Sampler        *sampler;
    ShaderProgram  *shader;
    Viewport        viewport;
    uint32          uniform_location;

    Members()
        : texture(NULL)
        , sampler(NULL)
        , shader(NULL)
        , uniform_location(0)
    {
    }

    istDefinePoolNewST(Members);
};
istMemberPtrImpl(EasyDrawState,Members);

EasyDrawState::EasyDrawState()
{
}

void EasyDrawState::setViewport( int32 width, int32 height )
{
    m->viewport = Viewport(ivec2(0,0), uvec2(width,height));
}
void EasyDrawState::setViewport( int32 x, int32 y, int32 width, int32 height )
{
    m->viewport = Viewport(ivec2(x,y), uvec2(width,height));
}
void EasyDrawState::setScreen(float32 width, float32 height)
{
    setScreen(0.0f, width, height, 0.0f);
}
void EasyDrawState::setScreen(float32 left, float32 right, float32 bottom, float32 top)
{
    m->proj = glm::ortho(left, right, bottom, top);
}
void EasyDrawState::setProjectionMatrix(const mat4 &mat){ m->proj=mat; }
void EasyDrawState::setWorldMatrix(const mat4 &mat)     { m->world=mat; }
void EasyDrawState::setTexture(Texture2D *tex)          { m->texture=tex; }
void EasyDrawState::setSampler(Sampler *smp)            { m->sampler=smp; }
void EasyDrawState::setShader(ShaderProgram *v)
{
    m->shader = v;
    if(m->shader) {
        m->uniform_location = m->shader->getUniformBlockIndex("render_states");
    }
}

const Viewport& EasyDrawState::getViewport() const          { return m->viewport; }
const mat4&     EasyDrawState::getProjectionMatrix() const  { return m->proj; }
const mat4&     EasyDrawState::getWorldMatrix() const       { return m->world; }
Texture2D*      EasyDrawState::getTexture() const           { return m->texture; }
Sampler*        EasyDrawState::getSampler() const           { return m->sampler; }
ShaderProgram*  EasyDrawState::getShader() const            { return m->shader; }
uint32          EasyDrawState::getUniformLocation() const   { return m->uniform_location; }


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
    I3D_TYPE index_type;
    uint32 num_vertices;
    size_t vb_offset; // in byte
    uint32 num_indices;
    size_t ib_offset; // in byte
};

struct EasyDrawer::Members
{
    typedef EasyDrawer::DrawCall DrawCall;

    VertexArray             *va;
    Buffer                  *vbo;
    Buffer                  *ibo;
    Buffer                  *ubo;
    EasyShaders             *shaders;
    ist::raw_vector<char>   vertex_data;
    ist::raw_vector<char>   index_data;
    ist::vector<DrawCall>   draw_calls;
    EasyDrawState           state;

    Members()
        : va(NULL)
        , vbo(NULL)
        , ibo(NULL)
        , ubo(NULL)
        , shaders(NULL)
    {
    }
};
istMemberPtrImpl(EasyDrawer,Members);


const EasyDrawState& EasyDrawer::getRenderStates() {return m->state; }
void EasyDrawer::forceSetRenderStates( const EasyDrawState &ds ) { m->state=ds; }

void EasyDrawer::setViewport( int32 width, int32 height )               { m->state.setViewport(width,height); }
void EasyDrawer::setViewport( int32 x, int32 y, int32 w, int32 h )      { m->state.setViewport(x,y, w,h);}
void EasyDrawer::setScreen(float32 width, float32 height)               { m->state.setScreen(width, height); }
void EasyDrawer::setScreen(float32 l, float32 r, float32 b, float32 t)  { m->state.setScreen(l,r, b,t); }
void EasyDrawer::setProjectionMatrix(const mat4 &mat)                   { m->state.setProjectionMatrix(mat); }
void EasyDrawer::setWorldMatrix(const mat4 &mat)                        { m->state.setWorldMatrix(mat); }
void EasyDrawer::setTexture(Texture2D *tex)                             { m->state.setTexture(tex); }
void EasyDrawer::setSampler(Sampler *smp)                               { m->state.setSampler(smp); }
void EasyDrawer::setShader(ShaderProgram *smp)                          { m->state.setShader(smp); }

const Viewport& EasyDrawer::getViewport() const         { return m->state.getViewport(); }
const mat4&     EasyDrawer::getProjectionMatrix() const { return m->state.getProjectionMatrix(); }
const mat4&     EasyDrawer::getWorldMatrix() const      { return m->state.getWorldMatrix(); }
Texture2D*      EasyDrawer::getTexture() const          { return m->state.getTexture(); }
Sampler*        EasyDrawer::getSampler() const          { return m->state.getSampler(); }
ShaderProgram*  EasyDrawer::getShader() const           { return m->state.getShader(); }
uint32          EasyDrawer::getUniformLocation() const  { return m->state.getUniformLocation(); }


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

template<class T> struct GetIndexType {};
template<> struct GetIndexType<uint8>  { static const I3D_TYPE result = I3D_UINT8; };
template<> struct GetIndexType<uint16> { static const I3D_TYPE result = I3D_UINT16; };
template<> struct GetIndexType<uint32> { static const I3D_TYPE result = I3D_UINT32; };

template<class VertexT>
void EasyDrawer::drawImpl( I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices )
{
    drawImpl<VertexT, uint8>(topology, vertices, num_vertices, NULL, 0);
}

template<class VertexT, class IndexT>
void EasyDrawer::drawImpl( I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const IndexT *indices, uint32 num_indices )
{
    if(num_vertices==0) { return; }

    VertexType vertex_type = GetVertexTypeID<VertexT>::result;
    DrawCall tmp_cmd = {
        m->state,
        topology,
        vertex_type,
        GetIndexType<IndexT>::result,
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

#define Template(VertexT)\
    void EasyDrawer::draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices) { drawImpl(topology, vertices, num_vertices); }\
    void EasyDrawer::draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint8  *indices, uint32 num_indices) { drawImpl(topology, vertices, num_vertices, indices, num_indices); }\
    void EasyDrawer::draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint16 *indices, uint32 num_indices) { drawImpl(topology, vertices, num_vertices, indices, num_indices); }\
    void EasyDrawer::draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint32 *indices, uint32 num_indices) { drawImpl(topology, vertices, num_vertices, indices, num_indices); }
istEachVertexTypes(Template)
#undef Template


void EasyDrawer::flush(DeviceContext *ctx)
{
    updateBuffers(ctx);

    mat4 prev_viewproj;
    for(size_t i=0; i<m->draw_calls.size(); ++i) {
        DrawCall &dc = m->draw_calls[i];
        const Viewport &vp = dc.state.getViewport();
        if(vp.getSize().x>0 && vp.getSize().y>0) {
            ctx->setViewport(vp);
        }

        mat4 viewproj = dc.state.getProjectionMatrix() * dc.state.getWorldMatrix();
        if(viewproj!=prev_viewproj) {
            prev_viewproj = viewproj;
            if(ctx->getDevice()->getSpec()->needs_transpose) {
                mat4 tmp = glm::transpose(viewproj);
                MapAndWrite(ctx, m->ubo, &tmp, sizeof(tmp));
            }
            else {
                MapAndWrite(ctx, m->ubo, &viewproj, sizeof(viewproj));
            }
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
            ctx->setIndexBuffer(m->ibo, dc.ib_offset, dc.index_type);
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
