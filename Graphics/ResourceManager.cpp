#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Light.h"
#include "GPGPU/SPH.cuh"
#include "shader/glsl_source.h"

namespace atomic {


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
    s_inst = IST_NEW(GraphicResourceManager) ();
    s_inst->initialize();
}

void GraphicResourceManager::finalizeInstance()
{
    s_inst->finalize();
    IST_SAFE_DELETE(s_inst);
}

inline AtomicShader* CreateAtomicShader(const char* source)
{
    AtomicShader *sh = IST_NEW(AtomicShader)();
    sh->initialize();
    sh->loadFromMemory(source);
    return sh;
}


bool GraphicResourceManager::initialize()
{
    m_font = NULL;
    stl::fill_n(m_tex2d, _countof(m_tex2d), (Texture2D*)NULL);
    stl::fill_n(m_va, _countof(m_va), (VertexArray*)NULL);
    stl::fill_n(m_vbo, _countof(m_vbo), (VertexBufferObject*)NULL);
    stl::fill_n(m_ibo, _countof(m_ibo), (IndexBufferObject*)NULL);
    stl::fill_n(m_ubo, _countof(m_ubo), (UniformBufferObject*)NULL);
    stl::fill_n(m_fbo, _countof(m_fbo), (FrameBufferObject*)NULL);
    stl::fill_n(m_shader, _countof(m_shader), (AtomicShader*)NULL);

    //// どうも 2 の n 乗サイズのフレームバッファの方が若干描画早いっぽい。 
    //uint32 framebuffer_width = atomicGetWindowWidth();
    //uint32 framebuffer_height = atomicGetWindowHeight();
    uint32 framebuffer_width = CalcFrameBufferWidth();
    uint32 framebuffer_height = CalcFrameBufferHeight();

    // initialize opengl resources
    {
        m_font = IST_NEW(SystemFont)();
        m_font->initialize();
    }
    {
        for(uint32 i=0; i<_countof(m_tex2d); ++i) {
            m_tex2d[i] = IST_NEW(Texture2D) ();
            m_tex2d[i]->initialize();
        }
    }
    {
        for(uint32 i=0; i<_countof(m_vbo); ++i) {
            m_vbo[i] = IST_NEW(VertexBufferObject) ();
            m_vbo[i]->initialize();
        }
    }
    {
        for(uint32 i=0; i<_countof(m_ibo); ++i) {
            m_ibo[i] = IST_NEW(IndexBufferObject) ();
            m_ibo[i]->initialize();
        }
    }
    {
        for(uint32 i=0; i<_countof(m_va); ++i) {
            m_va[i] = IST_NEW(VertexArray)();
            m_va[i]->initialize();
        }

        CreateScreenQuad(*m_va[VA_SCREEN_QUAD], *m_vbo[VBO_SCREEN_QUAD]);
        CreateBloomLuminanceQuads(*m_va[VA_BLOOM_LUMINANCE_QUADS], *m_vbo[VBO_BLOOM_LUMINANCE_QUADS]);
        CreateBloomBlurQuads(*m_va[VA_BLOOM_BLUR_QUADS], *m_vbo[VBO_BLOOM_BLUR_QUADS]);
        CreateBloomCompositeQuad(*m_va[VA_BLOOM_COMPOSITE_QUAD], *m_vbo[VBO_BLOOM_COMPOSITE_QUAD]);

        CreateCube(*m_va[VA_UNIT_CUBE], *m_vbo[VBO_UNIT_CUBE], 1.0f);
        CreateCube(*m_va[VA_FRACTION_CUBE], *m_vbo[VBO_FRACTION_CUBE], 0.03f);
        CreateSphere(*m_va[VA_UNIT_SPHERE], *m_vbo[VBO_UNIT_SPHERE], *m_ibo[IBO_SPHERE], 1.00f, 32,16);
    }
    {
        for(uint32 i=0; i<_countof(m_ubo); ++i) {
            m_ubo[i] = IST_NEW(UniformBufferObject) ();
            m_ubo[i]->initialize();
        }
        m_ubo[UBO_RENDER_STATES]->allocate(sizeof(RenderStates), UniformBufferObject::USAGE_DYNAMIC);
    }
    {
        // create shaders
        m_shader[SH_GBUFFER]            = CreateAtomicShader(g_GBuffer_Cube_glsl);
        m_shader[SH_POINTLIGHT]         = CreateAtomicShader(g_Deferred_PointLight_glsl);
        m_shader[SH_DIRECTIONALLIGHT]   = CreateAtomicShader(g_Deferred_DirectionalLight_glsl);
        m_shader[SH_BLOOM_LUMINANCE]    = CreateAtomicShader(g_Bloom_Luminance_glsl);
        m_shader[SH_BLOOM_HBLUR]        = CreateAtomicShader(g_Bloom_HBlur_glsl);
        m_shader[SH_BLOOM_VBLUR]        = CreateAtomicShader(g_Bloom_VBlur_glsl);
        m_shader[SH_BLOOM_COMPOSITE]    = CreateAtomicShader(g_Bloom_Composite_glsl);
        m_shader[SH_OUTPUT]             = CreateAtomicShader(g_Out_glsl);
    }
    {
        // create textures
        GenerateRandomTexture(*m_tex2d[TEX2D_RANDOM], 64, 64, Texture2D::FMT_RGB_U8);
    }
    {
        // create render targets
        m_rt_gbuffer = IST_NEW(RenderTargetGBuffer) ();
        m_rt_gbuffer->initialize(framebuffer_width, framebuffer_height, Color3DepthBuffer::FMT_RGBA_F32);

        m_rt_deferred = IST_NEW(RenderTargetDeferred) ();
        m_rt_deferred->setDepthBuffer(m_rt_gbuffer->getDepthBuffer());
        m_rt_deferred->initialize(framebuffer_width, framebuffer_height);

        for(uint32 i=0; i<_countof(m_rt_gauss); ++i) {
            m_rt_gauss[i] = IST_NEW(ColorBuffer) ();
            m_rt_gauss[i]->initialize(512, 256, ColorBuffer::FMT_RGBA_U8);
        }

        m_fbo[RT_GBUFFER] = m_rt_gbuffer;
        m_fbo[RT_DEFERRED] = m_rt_deferred;
        m_fbo[RT_GAUSS0] = m_rt_gauss[0];
        m_fbo[RT_GAUSS1] = m_rt_gauss[1];
    }

    m_vbo[VBO_FRACTION_INSTANCE]->allocate(sizeof(SPHParticle)*SPH_MAX_PARTICLE_NUM, VertexBufferObject::USAGE_DYNAMIC);
    m_vbo[VBO_DIRECTIONALLIGHT_INSTANCE]->allocate(sizeof(DirectionalLight)*ATOMIC_MAX_DIRECTIONAL_LIGHTS, VertexBufferObject::USAGE_DYNAMIC);
    m_vbo[VBO_POINTLIGHT_INSTANCE]->allocate(sizeof(PointLight)*ATOMIC_MAX_POINT_LIGHTS, VertexBufferObject::USAGE_DYNAMIC);
    SPHInitialize();
    SPHInitializeInstanceBuffers(m_vbo[VBO_FRACTION_INSTANCE]->getHandle(), m_vbo[VBO_POINTLIGHT_INSTANCE]->getHandle());

    return true;
}

void GraphicResourceManager::finalize()
{
    SPHFinalizeInstanceBuffers();
    SPHFinalize();

    if(m_font) { m_font->finalize(); IST_SAFE_DELETE(m_font); }
    for(uint32 i=0; i<_countof(m_tex2d); ++i)   { if(m_tex2d[i]) { m_tex2d[i]->finalize(); IST_SAFE_DELETE( m_tex2d[i] ); } }
    for(uint32 i=0; i<_countof(m_va); ++i)      { if(m_va[i]) { m_va[i]->finalize(); IST_SAFE_DELETE( m_va[i] ); } }
    for(uint32 i=0; i<_countof(m_vbo); ++i)     { if(m_vbo[i]) { m_vbo[i]->finalize(); IST_SAFE_DELETE( m_vbo[i] ); } }
    for(uint32 i=0; i<_countof(m_ibo); ++i)     { if(m_ibo[i]) { m_ibo[i]->finalize(); IST_SAFE_DELETE( m_ibo[i] ); } }
    for(uint32 i=0; i<_countof(m_ubo); ++i)     { if(m_ubo[i]) { m_ubo[i]->finalize(); IST_SAFE_DELETE( m_ubo[i] ); } }
    for(uint32 i=0; i<_countof(m_fbo); ++i)     { if(m_fbo[i]) { m_fbo[i]->finalize(); IST_SAFE_DELETE( m_fbo[i] ); } }
    for(uint32 i=0; i<_countof(m_shader); ++i)  { if(m_shader[i]) { m_shader[i]->finalize(); IST_SAFE_DELETE( m_shader[i] ); } }
}


} // namespace atomic
