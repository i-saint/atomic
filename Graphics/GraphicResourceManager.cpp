#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "GraphicResourceManager.h"
#include "../Game/AtomicApplication.h"

namespace atomic {


inline void SetFloat3(float (&v)[3], float x, float y, float z)
{
    v[0] = x;
    v[1] = y;
    v[2] = z;
}
inline void SetFloat3(float (&v)[3], float (&s)[3])
{
    v[0] = s[0];
    v[1] = s[1];
    v[2] = s[2];
}
inline void SetFloat3(float (&v)[3], XMVECTOR s)
{
    v[0] = ((float*)&s)[0];
    v[1] = ((float*)&s)[1];
    v[2] = ((float*)&s)[2];
}



inline void CreateSphereModel(ModelData& model, float32 radius)
{
    const float pi = 3.14159f;
    const float radian = pi/180.0f;

    const int ydiv = 12;
    const int xzdiv = 24;
    XMVECTOR v[ydiv][xzdiv];
    float n[ydiv][xzdiv][3];
    int index[(ydiv-1)*(xzdiv)*4];

    for(int i=0; i<ydiv; ++i) {
        float ang = ((180.0f/(ydiv-1)*i-90.0f)*radian);
        v[i][0] = XMVectorSet(cos(ang)*radius, sin(ang)*radius, 0, 1.0);
    }
    XMMATRIX mat = XMMatrixIdentity();
    for(int j=0; j<xzdiv; ++j) {
        for(int i=0; i<ydiv; ++i) {
            v[i][j] = XMVector4Transform(v[i][0], mat);
            SetFloat3(n[i][j], XMVector3Normalize(v[i][j]));
        }
        mat = XMMatrixRotationY(360.0f/xzdiv*j*radian);
    }

    int *ci = index;
    for(int i=0; i<ydiv-1; ++i) {
        for(int j=0; j<xzdiv; ++j) {
            ci[0] = xzdiv*(i)  + j;
            ci[1] = xzdiv*(i)  + ((j+1)%xzdiv);
            ci[2] = xzdiv*(i+1)+ ((j+1)%xzdiv);
            ci[3] = xzdiv*(i+1)+ j;
            ci+=4;
        }
    }
    model.setVertex(v, ydiv*xzdiv, ModelData::VTX_FLOAT4, ModelData::USAGE_STATIC);
    model.setNormal(n, ydiv*xzdiv, ModelData::USAGE_STATIC);
    model.setIndex(index, ((ydiv-1)*(xzdiv)*4), ModelData::IDX_INT32, ModelData::PRM_QUADS, ModelData::USAGE_STATIC);

}

inline void CreateCubeModel(ModelData& model, float32 len)
{
    float vertex[24][3];
    float normal[24][3];
    int index[24];

    float n[3];
    float ur[3];
    float bl[3];
    SetFloat3(ur,  len/2.0f, len/2.0f, len/2.0f);
    SetFloat3(bl, -len/2.0f,-len/2.0f,-len/2.0f);

    SetFloat3(n, 1.0f, 0.0f, 0.0f);
    SetFloat3(normal[0], n);
    SetFloat3(normal[1], n);
    SetFloat3(normal[2], n);
    SetFloat3(normal[3], n);
    SetFloat3(vertex[0], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[1], ur[0], bl[1], ur[2]);
    SetFloat3(vertex[2], ur[0], bl[1], bl[2]);
    SetFloat3(vertex[3], ur[0], ur[1], bl[2]);

    SetFloat3(n, -1.0f, 0.0f, 0.0f);
    SetFloat3(normal[4], n);
    SetFloat3(normal[5], n);
    SetFloat3(normal[6], n);
    SetFloat3(normal[7], n);
    SetFloat3(vertex[4], bl[0], ur[1], ur[2]);
    SetFloat3(vertex[5], bl[0], ur[1], bl[2]);
    SetFloat3(vertex[6], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[7], bl[0], bl[1], ur[2]);

    SetFloat3(n, 0.0f, 1.0f, 0.0f);
    SetFloat3(normal[8], n);
    SetFloat3(normal[9], n);
    SetFloat3(normal[10], n);
    SetFloat3(normal[11], n);
    SetFloat3(vertex[8], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[9], ur[0], ur[1], bl[2]);
    SetFloat3(vertex[10], bl[0], ur[1], bl[2]);
    SetFloat3(vertex[11], bl[0], ur[1], ur[2]);

    SetFloat3(n, 0.0f, -1.0f, 0.0f);
    SetFloat3(normal[12], n);
    SetFloat3(normal[13], n);
    SetFloat3(normal[14], n);
    SetFloat3(normal[15], n);
    SetFloat3(vertex[12], ur[0], bl[1], ur[2]);
    SetFloat3(vertex[13], bl[0], bl[1], ur[2]);
    SetFloat3(vertex[14], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[15], ur[0], bl[1], bl[2]);

    SetFloat3(n, 0.0f, 0.0f, 1.0f);
    SetFloat3(normal[16], n);
    SetFloat3(normal[17], n);
    SetFloat3(normal[18], n);
    SetFloat3(normal[19], n);
    SetFloat3(vertex[16], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[17], bl[0], ur[1], ur[2]);
    SetFloat3(vertex[18], bl[0], bl[1], ur[2]);
    SetFloat3(vertex[19], ur[0], bl[1], ur[2]);

    SetFloat3(n, 0.0f, 0.0f, -1.0f);
    SetFloat3(normal[20], n);
    SetFloat3(normal[21], n);
    SetFloat3(normal[22], n);
    SetFloat3(normal[23], n);
    SetFloat3(vertex[20], ur[0], ur[1], bl[2]);
    SetFloat3(vertex[21], ur[0], bl[1], bl[2]);
    SetFloat3(vertex[22], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[23], bl[0], ur[1], bl[2]);

    for(size_t i=0; i<24; ++i) {
        index[i] = i;
    }

    model.setVertex(vertex, 24, ModelData::VTX_FLOAT3, ModelData::USAGE_STATIC);
    model.setNormal(normal, 24, ModelData::USAGE_STATIC);
    model.setIndex(index, 24, ModelData::IDX_INT32, ModelData::PRM_QUADS, ModelData::USAGE_STATIC);
}

void DrawScreen(float32 min_tx, float32 min_ty, float32 max_tx, float32 max_ty)
{
    OrthographicCamera cam;
    cam.setScreen(0.0f, 1.0f, 0.0f, 1.0f);
    cam.bind();

    float32 min_x = 0.0f;
    float32 min_y = 0.0f;
    float32 max_x = 1.0f;
    float32 max_y = 1.0f;

    glBegin(GL_QUADS);
    glTexCoord2f(min_tx, min_ty);
    glVertex2f(min_x, min_y);
    glTexCoord2f(max_tx, min_ty);
    glVertex2f(max_x, min_y);
    glTexCoord2f(max_tx, max_ty);
    glVertex2f(max_x, max_y);
    glTexCoord2f(min_tx, max_ty);
    glVertex2f(min_x, max_y);
    glEnd();
}


uint32 CalcFrameBufferWidth()
{
    uint32 r = 256;
    uint32 window_width = GetWindowWidth();
    while(r < window_width) {
        r *= 2;
    }
    return r;
}

uint32 CalcFrameBufferHeight()
{
    uint32 r = 256;
    uint32 window_height = GetWindowHeight();
    while(r < window_height) {
        r *= 2;
    }
    return r;
}


GraphicResourceManager* GraphicResourceManager::s_inst = NULL;

void GraphicResourceManager::intializeInstance()
{
    s_inst = AT_NEW(GraphicResourceManager) GraphicResourceManager();
    s_inst->initialize();
}

void GraphicResourceManager::finalizeInstance()
{
    s_inst->finalize();
    AT_DELETE(s_inst)
}


bool GraphicResourceManager::initialize()
{
    stl::fill_n(m_model, _countof(m_model), (ModelData*)NULL);
    stl::fill_n(m_tex2d, _countof(m_tex2d), (Texture2D*)NULL);
    stl::fill_n(m_vbo, _countof(m_vbo), (VertexBufferObject*)NULL);
    stl::fill_n(m_ubo, _countof(m_ubo), (UniformBufferObject*)NULL);
    stl::fill_n(m_fbo, _countof(m_fbo), (FrameBufferObject*)NULL);
    stl::fill_n(m_shader, _countof(m_shader), (ProgramObject*)NULL);

    uint32 framebuffer_width = CalcFrameBufferWidth();
    uint32 framebuffer_height = CalcFrameBufferHeight();

    {
        for(uint32 i=0; i<_countof(m_model); ++i) {
            m_model[i] = AT_NEW(ModelData) ModelData();
            m_model[i]->initialize();
        }
        CreateCubeModel(*m_model[MODEL_CUBE], 6.0f);
        CreateSphereModel(*m_model[MODEL_SPHERE], 150.0f);
    }
    {
        for(uint32 i=0; i<_countof(m_tex2d); ++i) {
            m_tex2d[i] = AT_NEW(Texture2D) Texture2D();
            m_tex2d[i]->initialize();
        }
    }
    {
        for(uint32 i=0; i<_countof(m_vbo); ++i) {
            m_vbo[i] = AT_NEW(VertexBufferObject) VertexBufferObject();
            m_vbo[i]->initialize();
        }
    }
    {
        for(uint32 i=0; i<_countof(m_ubo); ++i) {
            m_ubo[i] = AT_NEW(UniformBufferObject) UniformBufferObject();
            m_ubo[i]->initialize();
        }
    }
    {
        m_sh_gbuffer    = AT_NEW(ShaderGBuffer) ShaderGBuffer();
        m_sh_deferred   = AT_NEW(ShaderDeferred) ShaderDeferred();
        m_sh_output     = AT_NEW(ShaderOutput) ShaderOutput();
        m_sh_gbuffer->initialize();
        m_sh_deferred->initialize();
        m_sh_output->initialize();
        m_shader[SH_GBUFFER]    = m_sh_gbuffer;
        m_shader[SH_DEFERRED]   = m_sh_deferred;
        m_shader[SH_OUTPUT]     = m_sh_output;
    }
    {
        //m_rand.initialize(0);
        //m_tex_rand.initialize(64, 64, Texture2D::FMT_RGB_U8, m_rand);
    }
    {
        m_rt_gbuffer = AT_NEW(RenderTargetGBuffer) RenderTargetGBuffer();
        m_rt_gbuffer->initialize(framebuffer_width, framebuffer_height, Color3DepthBuffer::FMT_RGBA_F32);

        m_rt_deferred = AT_NEW(RenderTargetDeferred) RenderTargetDeferred();
        m_rt_deferred->setDepthBuffer(m_rt_gbuffer->getDepthBuffer());
        m_rt_deferred->initialize(framebuffer_width, framebuffer_height);

        for(uint32 i=0; i<_countof(m_rt_gauss); ++i) {
            m_rt_gauss[i] = AT_NEW(ColorBuffer) ColorBuffer();
            m_rt_gauss[i]->initialize(512, 512, ColorBuffer::FMT_RGBA_U8);
        }

        m_fbo[RT_GBUFFER] = m_rt_gbuffer;
        m_fbo[RT_DEFERRED] = m_rt_deferred;
        m_fbo[RT_GAUSS0] = m_rt_gauss[0];
        m_fbo[RT_GAUSS1] = m_rt_gauss[1];
    }

    return true;
}

void GraphicResourceManager::finalize()
{
    for(uint32 i=0; i<_countof(m_model); ++i)   { if(m_model[i]) m_model[i]->finalize(); AT_DELETE( m_model[i] ); }
    for(uint32 i=0; i<_countof(m_tex2d); ++i)   { if(m_tex2d[i]) m_tex2d[i]->finalize(); AT_DELETE( m_tex2d[i] ); }
    for(uint32 i=0; i<_countof(m_vbo); ++i)     { if(m_vbo[i]) m_vbo[i]->finalize(); AT_DELETE( m_vbo[i] ); }
    for(uint32 i=0; i<_countof(m_ubo); ++i)     { if(m_ubo[i]) m_ubo[i]->finalize(); AT_DELETE( m_ubo[i] ); }
    for(uint32 i=0; i<_countof(m_fbo); ++i)     { if(m_fbo[i]) m_fbo[i]->finalize(); AT_DELETE( m_fbo[i] ); }
    for(uint32 i=0; i<_countof(m_shader); ++i)  { if(m_shader[i]) m_shader[i]->finalize(); AT_DELETE( m_shader[i] ); }
}


} // namespace atomic
