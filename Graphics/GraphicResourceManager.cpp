#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "GraphicResourceManager.h"
#include "../Game/AtomicApplication.h"

namespace atomic {



inline void CreateSphereModel(ModelData& model, float32 radius)
{
    const float pi = 3.14159f;
    const float radian = pi/180.0f;

    const int ydiv = 12;
    const int xzdiv = 24;
    vec4 v[ydiv][xzdiv];
    vec3 n[ydiv][xzdiv];
    int index[(ydiv-1)*(xzdiv)*4];

    for(int i=0; i<ydiv; ++i) {
        float ang = ((180.0f/(ydiv-1)*i-90.0f)*radian);
        v[i][0] = vec4(cos(ang)*radius, sin(ang)*radius, 0, 1.0);
    }

    mat4 mat;
    for(int j=0; j<xzdiv; ++j) {
        for(int i=0; i<ydiv; ++i) {
            v[i][j] = mat * v[i][0];
            n[i][j] = glm::normalize(vec3(v[i][j].x, v[i][j].y, v[i][j].z));
        }
        mat = glm::rotate(mat4(), 360.0f/xzdiv*j, vec3(0.0f, 1.0f, 0.0f));
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
    model.setData(0, v, ydiv*xzdiv, 4);
    model.setData(1, n, ydiv*xzdiv, 3);
    model.setIndex(index, ((ydiv-1)*(xzdiv)*4), ModelData::IDX_INT32, ModelData::PRM_QUADS);

}

inline void CreateCubeModel(ModelData& model, float32 len)
{
    vec4 vertex[24];
    vec3 normal[24];
    int index[24];

    vec3 ur = vec3( len/2.0f, len/2.0f, len/2.0f);
    vec3 bl = vec3(-len/2.0f,-len/2.0f,-len/2.0f);
    vec3 n;

    n = vec3(1.0f, 0.0f, 0.0f);
    normal[0] = n;
    normal[1] = n;
    normal[2] = n;
    normal[3] = n;
    vertex[0] = vec4(ur[0], ur[1], ur[2], 1.0f);
    vertex[1] = vec4(ur[0], bl[1], ur[2], 1.0f);
    vertex[2] = vec4(ur[0], bl[1], bl[2], 1.0f);
    vertex[3] = vec4(ur[0], ur[1], bl[2], 1.0f);

    n = vec3(-1.0f, 0.0f, 0.0f);
    normal[4] = n;
    normal[5] = n;
    normal[6] = n;
    normal[7] = n;
    vertex[4] = vec4(bl[0], ur[1], ur[2], 1.0f);
    vertex[5] = vec4(bl[0], ur[1], bl[2], 1.0f);
    vertex[6] = vec4(bl[0], bl[1], bl[2], 1.0f);
    vertex[7] = vec4(bl[0], bl[1], ur[2], 1.0f);

    n = vec3(0.0f, 1.0f, 0.0f);
    normal[8] = n;
    normal[9] = n;
    normal[10] = n;
    normal[11] = n;
    vertex[8] = vec4(ur[0], ur[1], ur[2], 1.0f);
    vertex[9] = vec4(ur[0], ur[1], bl[2], 1.0f);
    vertex[10] = vec4(bl[0], ur[1], bl[2], 1.0f);
    vertex[11] = vec4(bl[0], ur[1], ur[2], 1.0f);

    n = vec3(0.0f, -1.0f, 0.0f);
    normal[12] = n;
    normal[13] = n;
    normal[14] = n;
    normal[15] = n;
    vertex[12] = vec4(ur[0], bl[1], ur[2], 1.0f);
    vertex[13] = vec4(bl[0], bl[1], ur[2], 1.0f);
    vertex[14] = vec4(bl[0], bl[1], bl[2], 1.0f);
    vertex[15] = vec4(ur[0], bl[1], bl[2], 1.0f);

    n = vec3(0.0f, 0.0f, 1.0f);
    normal[16] = n;
    normal[17] = n;
    normal[18] = n;
    normal[19] = n;
    vertex[16] = vec4(ur[0], ur[1], ur[2], 1.0f);
    vertex[17] = vec4(bl[0], ur[1], ur[2], 1.0f);
    vertex[18] = vec4(bl[0], bl[1], ur[2], 1.0f);
    vertex[19] = vec4(ur[0], bl[1], ur[2], 1.0f);

    n = vec3(0.0f, 0.0f, -1.0f);
    normal[20] = n;
    normal[21] = n;
    normal[22] = n;
    normal[23] = n;
    vertex[20] = vec4(ur[0], ur[1], bl[2], 1.0f);
    vertex[21] = vec4(ur[0], bl[1], bl[2], 1.0f);
    vertex[22] = vec4(bl[0], bl[1], bl[2], 1.0f);
    vertex[23] = vec4(bl[0], ur[1], bl[2], 1.0f);

    for(size_t i=0; i<24; ++i) {
        index[i] = i;
    }

    model.setData(0, &vertex, 24, 4);
    model.setData(1, normal, 24, 3);
    model.setIndex(index, 24, ModelData::IDX_INT32, ModelData::PRM_QUADS);
}

void DrawScreen(vec2 min_pos, vec2 max_pos, vec2 min_tc, vec2 max_tc)
{
    glBegin(GL_QUADS);
    glTexCoord2f(min_tc.x, min_tc.y);
    glVertex2f(min_pos.x, min_pos.y);
    glTexCoord2f(max_tc.x, min_tc.y);
    glVertex2f(max_pos.x, min_pos.y);
    glTexCoord2f(max_tc.x, max_tc.y);
    glVertex2f(max_pos.x, max_pos.y);
    glTexCoord2f(min_tc.x, max_tc.y);
    glVertex2f(min_pos.x, max_pos.y);
    glEnd();
}

void DrawScreen(vec2 min_tc, vec2 max_tc)
{
    DrawScreen(vec2(0.0f, 0.0f), vec2(1.0f, 1.0f), min_tc, max_tc);
}

void DrawScreen()
{
    DrawScreen(vec2(0.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 0.0f), vec2(1.0f, 1.0f));
}


uint32 CalcFrameBufferWidth()
{
    uint32 r = 256;
    uint32 window_width = atomicGetWindowWidth();
    while(r < window_width) {
        r *= 2;
    }
    return r;
}

uint32 CalcFrameBufferHeight()
{
    uint32 r = 256;
    uint32 window_height = atomicGetWindowHeight();
    while(r < window_height) {
        r *= 2;
    }
    return r;
}


GraphicResourceManager* GraphicResourceManager::s_inst = NULL;

void GraphicResourceManager::intializeInstance()
{
    s_inst = AT_NEW(GraphicResourceManager) ();
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
            m_model[i] = AT_NEW(ModelData) ();
            m_model[i]->initialize();
        }
        CreateCubeModel(*m_model[MODEL_CUBE], 6.0f);
        CreateSphereModel(*m_model[MODEL_SPHERE], 150.0f);
    }
    {
        for(uint32 i=0; i<_countof(m_tex2d); ++i) {
            m_tex2d[i] = AT_NEW(Texture2D) ();
            m_tex2d[i]->initialize();
        }
    }
    {
        for(uint32 i=0; i<_countof(m_vbo); ++i) {
            m_vbo[i] = AT_NEW(VertexBufferObject) ();
            m_vbo[i]->initialize();
        }
    }
    {
        for(uint32 i=0; i<_countof(m_ubo); ++i) {
            m_ubo[i] = AT_NEW(UniformBufferObject) ();
            m_ubo[i]->initialize();
        }
    }
    {
        m_sh_gbuffer    = AT_NEW(ShaderGBuffer) ();
        m_sh_deferred   = AT_NEW(ShaderDeferred) ();
        m_sh_bloom      = AT_NEW(ShaderBloom) ();
        m_sh_output     = AT_NEW(ShaderOutput) ();
        m_sh_gbuffer->initialize();
        m_sh_deferred->initialize();
        m_sh_bloom->initialize();
        m_sh_output->initialize();
        m_shader[SH_GBUFFER]    = m_sh_gbuffer;
        m_shader[SH_DEFERRED]   = m_sh_deferred;
        m_shader[SH_BLOOM]      = m_sh_bloom;
        m_shader[SH_OUTPUT]     = m_sh_output;
    }
    {
        //m_rand.initialize(0);
        //m_tex_rand.initialize(64, 64, Texture2D::FMT_RGB_U8, m_rand);
    }
    {
        m_rt_gbuffer = AT_NEW(RenderTargetGBuffer) ();
        m_rt_gbuffer->initialize(framebuffer_width, framebuffer_height, Color3DepthBuffer::FMT_RGBA_F32);

        m_rt_deferred = AT_NEW(RenderTargetDeferred) ();
        m_rt_deferred->setDepthBuffer(m_rt_gbuffer->getDepthBuffer());
        m_rt_deferred->initialize(framebuffer_width, framebuffer_height);

        for(uint32 i=0; i<_countof(m_rt_gauss); ++i) {
            m_rt_gauss[i] = AT_NEW(ColorBuffer) ();
            m_rt_gauss[i]->initialize(512, 256, ColorBuffer::FMT_RGBA_U8);
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
