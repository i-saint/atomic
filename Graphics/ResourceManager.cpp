#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Graphics/AtomicRenderingSystem.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Light.h"
#include "shader/glsl_source.h"

namespace atomic {

uvec2 CalcFrameBufferSize()
{
    uvec2 r = uvec2(128);
    uvec2 wsize = atomicGetWindowSize();
    while(r.x < wsize.x) { r.x *= 2; }
    while(r.y < wsize.y) { r.y *= 2; }
    return r;
}


GraphicResourceManager* GraphicResourceManager::s_inst = NULL;

void GraphicResourceManager::intializeInstance()
{
    s_inst = istNew(GraphicResourceManager)();
    s_inst->initialize();
}

void GraphicResourceManager::finalizeInstance()
{
    s_inst->finalize();
    istSafeDelete(s_inst);
}

inline AtomicShader* CreateAtomicShader(const char* source)
{
    AtomicShader *sh = istNew(AtomicShader)();
    sh->loadFromMemory(source);
    return sh;
}


bool GraphicResourceManager::initialize()
{
    m_font = NULL;
    stl::fill_n(m_sampler, _countof(m_sampler), (Sampler*)NULL);
    stl::fill_n(m_tex1d, _countof(m_tex1d), (Texture1D*)NULL);
    stl::fill_n(m_tex2d, _countof(m_tex2d), (Texture2D*)NULL);
    stl::fill_n(m_va, _countof(m_va), (VertexArray*)NULL);
    stl::fill_n(m_vbo, _countof(m_vbo), (Buffer*)NULL);
    stl::fill_n(m_ibo, _countof(m_ibo), (Buffer*)NULL);
    stl::fill_n(m_ubo, _countof(m_ubo), (Buffer*)NULL);
    stl::fill_n(m_rt, _countof(m_rt), (RenderTarget*)NULL);
    stl::fill_n(m_shader, _countof(m_shader), (AtomicShader*)NULL);

    //// どうも 2 の n 乗サイズのフレームバッファの方が若干描画早いっぽい。 
    uvec2 rt_size = atomicGetWindowSize();
    //uvec2 rt_size = CalcFrameBufferSize();

    // initialize opengl resources
    i3d::Device *dev = atomicGetGLDevice();
    {
        m_font = istNew(SystemFont)(dev->getHDC());
    }
    for(uint32 i=0; i<_countof(m_va); ++i) {
        m_va[i] = dev->createVertexArray();
    }

    {
        CreateFloorQuad(m_va[VA_FLOOR_QUAD], m_vbo[VBO_FLOOR_QUAD], vec4(-SPH_GRID_SIZE*0.5f, -SPH_GRID_SIZE*0.5f, -0.15f, 0.0f), vec4(SPH_GRID_SIZE, SPH_GRID_SIZE, 0.0f, 0.0f));
        CreateScreenQuad(m_va[VA_SCREEN_QUAD], m_vbo[VBO_SCREEN_QUAD]);
        CreateBloomLuminanceQuads(m_va[VA_BLOOM_LUMINANCE_QUADS], m_vbo[VBO_BLOOM_LUMINANCE_QUADS]);
        CreateBloomBlurQuads(m_va[VA_BLOOM_BLUR_QUADS], m_vbo[VBO_BLOOM_BLUR_QUADS]);
        CreateBloomCompositeQuad(m_va[VA_BLOOM_COMPOSITE_QUAD], m_vbo[VBO_BLOOM_COMPOSITE_QUAD]);

        CreateCube(m_va[VA_UNIT_CUBE], m_vbo[VBO_UNIT_CUBE], 0.5f);
        CreateCube(m_va[VA_FLUID_CUBE], m_vbo[VBO_FLUID_CUBE], 0.015f);
        CreateSphere(m_va[VA_UNIT_SPHERE], m_vbo[VBO_UNIT_SPHERE], m_ibo[IBO_LIGHT_SPHERE], 1.00f, 32,16);
        CreateSphere(m_va[VA_BLOOSTAIN_SPHERE], m_vbo[VBO_BLOODSTAIN_SPHERE], m_ibo[IBO_BLOODSTAIN_SPHERE], 0.075f, 8,8);

        CreateFieldGridLines(m_va[VA_FIELD_GRID], m_vbo[VBO_FIELD_GRID]);
        CreateDistanceFieldQuads(m_va[VA_DISTANCE_FIELD],
            m_vbo[VBO_DISTANCE_FIELD_QUAD], m_vbo[VBO_DISTANCE_FIELD_POS], m_vbo[VBO_DISTANCE_FIELD_DIST]);

        m_vbo[VBO_FLUID_PARTICLES] = CreateVertexBuffer(dev, sizeof(sphFluidParticle)*SPH_MAX_FLUID_PARTICLES, I3D_USAGE_DYNAMIC);
        m_vbo[VBO_RIGID_PARTICLES] = CreateVertexBuffer(dev, sizeof(PSetParticle)*SPH_MAX_RIGID_PARTICLES, I3D_USAGE_DYNAMIC);
        m_vbo[VBO_PARTICLES] = CreateVertexBuffer(dev, sizeof(IndivisualParticle)*SPH_MAX_INDIVISUAL_PARTICLES, I3D_USAGE_DYNAMIC);
        m_vbo[VBO_DIRECTIONALLIGHT_INSTANCES] = CreateVertexBuffer(dev, sizeof(DirectionalLight)*ATOMIC_MAX_DIRECTIONAL_LIGHTS, I3D_USAGE_DYNAMIC);
        m_vbo[VBO_POINTLIGHT_INSTANCES] = CreateVertexBuffer(dev, sizeof(PointLight)*ATOMIC_MAX_POINT_LIGHTS, I3D_USAGE_DYNAMIC);
        m_vbo[VBO_BLOODSTAIN_PARTICLES] = CreateVertexBuffer(dev, sizeof(BloodstainParticle)*SPH_MAX_FLUID_PARTICLES, I3D_USAGE_DYNAMIC);
    }
    {
        m_ubo[UBO_RENDER_STATES]            = CreateUniformBuffer(dev, sizeof(RenderStates), I3D_USAGE_DYNAMIC);
        m_ubo[UBO_FXAA_PARAMS]              = CreateUniformBuffer(dev, sizeof(FXAAParams), I3D_USAGE_DYNAMIC);
        m_ubo[UBO_FADE_PARAMS]              = CreateUniformBuffer(dev, sizeof(FadeParams), I3D_USAGE_DYNAMIC);
        m_ubo[UBO_FILL_PARAMS]              = CreateUniformBuffer(dev, sizeof(FillParams), I3D_USAGE_DYNAMIC);
        m_ubo[UBO_MULTIRESOLUTION_PARAMS]   = CreateUniformBuffer(dev, sizeof(MultiresolutionParams), I3D_USAGE_DYNAMIC);
        m_ubo[UBO_DEBUG_SHOW_BUFFER_PARAMS] = CreateUniformBuffer(dev, sizeof(DebugShowBufferParams), I3D_USAGE_DYNAMIC);
    }
    {
        // create shaders
        m_shader[SH_GBUFFER_FLOOR]      = CreateAtomicShader(g_GBuffer_Floor_glsl);
        //m_shader[SH_GBUFFER_FLUID]      = CreateAtomicShader(g_GBuffer_Fluid_glsl);
        //m_shader[SH_GBUFFER_RIGID]      = CreateAtomicShader(g_GBuffer_Rigid_glsl);
        m_shader[SH_GBUFFER_FLUID]      = CreateAtomicShader(g_GBuffer_FluidBlood_glsl);
        //m_shader[SH_GBUFFER_FLUID]      = CreateAtomicShader(g_GBuffer_FluidSpherical_glsl);
        m_shader[SH_GBUFFER_RIGID]      = CreateAtomicShader(g_GBuffer_RigidSpherical_glsl);
        m_shader[SH_GBUFFER_PARTICLES]  = CreateAtomicShader(g_GBuffer_ParticleSpherical_glsl);
        m_shader[SH_BLOODSTAIN]         = CreateAtomicShader(g_Deferred_Bloodstain_glsl);
        m_shader[SH_UPSAMPLING]         = CreateAtomicShader(g_Deferred_Upsampling_glsl);
        m_shader[SH_POINTLIGHT]         = CreateAtomicShader(g_Deferred_PointLight_glsl);
        m_shader[SH_DIRECTIONALLIGHT]   = CreateAtomicShader(g_Deferred_DirectionalLight_glsl);
        m_shader[SH_MICROSCOPIC]        = CreateAtomicShader(g_Postprocess_Microscopic_glsl);
        m_shader[SH_FXAA_LUMA]          = CreateAtomicShader(g_FXAA_luma_glsl);
        m_shader[SH_FXAA]               = CreateAtomicShader(g_FXAA_glsl);
        m_shader[SH_BLOOM_LUMINANCE]    = CreateAtomicShader(g_Bloom_Luminance_glsl);
        m_shader[SH_BLOOM_HBLUR]        = CreateAtomicShader(g_Bloom_HBlur_glsl);
        m_shader[SH_BLOOM_VBLUR]        = CreateAtomicShader(g_Bloom_VBlur_glsl);
        m_shader[SH_BLOOM_COMPOSITE]    = CreateAtomicShader(g_Bloom_Composite_glsl);
        m_shader[SH_FADE]               = CreateAtomicShader(g_Fade_glsl);
        m_shader[SH_FILL]               = CreateAtomicShader(g_Fill_glsl);
        m_shader[SH_FILL_INSTANCED]     = CreateAtomicShader(g_FillInstanced_glsl);
        m_shader[SH_DISTANCE_FIELD]     = CreateAtomicShader(g_DistanceField_glsl);
        m_shader[SH_OUTPUT]             = CreateAtomicShader(g_Out_glsl);
        m_shader[SH_DEBUG_SHOW_RGB]     = CreateAtomicShader(g_Debug_ShowRGB_glsl);
        m_shader[SH_DEBUG_SHOW_AAA]     = CreateAtomicShader(g_Debug_ShowAAA_glsl);
    }
    {
        // samplers
        m_sampler[SAMPLER_GBUFFER]          = dev->createSampler(SamplerDesc(I3D_REPEAT, I3D_REPEAT, I3D_REPEAT, I3D_NEAREST, I3D_NEAREST));
        m_sampler[SAMPLER_TEXTURE_DEFAULT]  = dev->createSampler(SamplerDesc(I3D_REPEAT, I3D_REPEAT, I3D_REPEAT, I3D_LINEAR, I3D_LINEAR));
    }
    {
        // create textures
        m_tex2d[TEX2D_RANDOM] = GenerateRandomTexture(dev, uvec2(64, 64), I3D_RGB8U);
        m_tex2d[TEX2D_ENTITY_PARAMS] = dev->createTexture2D(Texture2DDesc(I3D_RGBA32F, uvec2(4, 4096*2)));
    }
    {
        // create render targets
        m_rt[RT_GBUFFER]    = i3d::CreateRenderTarget(dev, 4, rt_size, I3D_RGBA16F, I3D_DEPTH24_STENCIL8, 3, 3);
        m_rt[RT_GAUSS0]     = i3d::CreateRenderTarget(dev, 1, uvec2(512, 256), I3D_RGBA8U);
        m_rt[RT_GAUSS1]     = i3d::CreateRenderTarget(dev, 1, uvec2(512, 256), I3D_RGBA8U);
        m_rt[RT_OUTPUT0]    = i3d::CreateRenderTarget(dev, 1, rt_size, I3D_RGBA8U);
        m_rt[RT_OUTPUT1]    = i3d::CreateRenderTarget(dev, 1, rt_size, I3D_RGBA8U);
        m_rt[RT_OUTPUT_HALF]    = i3d::CreateRenderTarget(dev, 1, rt_size/uvec2(2,2), I3D_RGBA8U);
        m_rt[RT_OUTPUT_QUARTER] = i3d::CreateRenderTarget(dev, 1, rt_size/uvec2(4,4), I3D_RGBA8U);

        m_rt[RT_GENERIC]    = i3d::CreateRenderTarget(dev, 0, rt_size, I3D_RGBA8U);
    }

    {
        sphParams sph_params;
        sph_params.smooth_len           = 0.02f;
        sph_params.pressure_stiffness   = 50.0f;
        sph_params.rest_density         = 500.0f;
        sph_params.particle_mass        = 0.001f;
        sph_params.viscosity            = 0.2f;
        sph_params.density_coef         = sph_params.particle_mass * 315.0f / (64.0f * CUDART_PI_F * pow(sph_params.smooth_len, 9));
        sph_params.grad_pressure_coef   = sph_params.particle_mass * -45.0f / (CUDART_PI_F * pow(sph_params.smooth_len, 6));
        sph_params.lap_viscosity_coef   = sph_params.particle_mass * sph_params.viscosity * 45.0f / (CUDART_PI_F * pow(sph_params.smooth_len, 6));
        sph_params.wall_stiffness       = 3000.0f;
        SPHInitialize(sph_params);
        SPHInitializeGLBuffers( m_vbo[VBO_FLUID_PARTICLES]->getHandle() );
    }
    {
        CreateCubeParticleSet(m_pset[PSET_CUBE_SMALL],  m_rinfo[PSET_CUBE_SMALL],  0.1f);
        CreateCubeParticleSet(m_pset[PSET_CUBE_MEDIUM], m_rinfo[PSET_CUBE_MEDIUM], 0.2f);
        CreateCubeParticleSet(m_pset[PSET_CUBE_LARGE],  m_rinfo[PSET_CUBE_LARGE],  0.4f);
        CreateSphereParticleSet(m_pset[PSET_SPHERE_SMALL],  m_rinfo[PSET_SPHERE_SMALL],  0.125f);
        CreateSphereParticleSet(m_pset[PSET_SPHERE_MEDIUM], m_rinfo[PSET_SPHERE_MEDIUM], 0.25f);
        CreateSphereParticleSet(m_pset[PSET_SPHERE_LARGE],  m_rinfo[PSET_SPHERE_LARGE],  0.5f);
        CreateBulletParticleSet(m_pset[PSET_SPHERE_BULLET], m_rinfo[PSET_SPHERE_BULLET]);
    }


    {
        Sampler *smp_gb = atomicGetSampler(SAMPLER_GBUFFER);
        Sampler *smp_tex = atomicGetSampler(SAMPLER_TEXTURE_DEFAULT);
        smp_tex->bind(GLSL_COLOR_BUFFER);
        smp_gb->bind(GLSL_NORMAL_BUFFER);
        smp_gb->bind(GLSL_POSITION_BUFFER);
        smp_tex->bind(GLSL_GLOW_BUFFER);
        smp_tex->bind(GLSL_BACK_BUFFER);
        smp_tex->bind(GLSL_RANDOM_BUFFER);
        smp_tex->bind(GLSL_PARAM_BUFFER);
   }
    return true;
}

void GraphicResourceManager::finalize()
{
    SPHFinalizeGLBuffers();
    SPHFinalize();

    for(uint32 i=0; i<_countof(m_shader); ++i)  { if(m_shader[i]) { atomicSafeRelease( m_shader[i] ); } }
    for(uint32 i=0; i<_countof(m_rt); ++i)      { if(m_rt[i]) { atomicSafeRelease( m_rt[i] ); } }
    for(uint32 i=0; i<_countof(m_ubo); ++i)     { if(m_ubo[i]) { atomicSafeRelease( m_ubo[i] ); } }
    for(uint32 i=0; i<_countof(m_ibo); ++i)     { if(m_ibo[i]) { atomicSafeRelease( m_ibo[i] ); } }
    for(uint32 i=0; i<_countof(m_vbo); ++i)     { if(m_vbo[i]) { atomicSafeRelease( m_vbo[i] ); } }
    for(uint32 i=0; i<_countof(m_va); ++i)      { if(m_va[i]) { atomicSafeRelease( m_va[i] ); } }
    for(uint32 i=0; i<_countof(m_tex2d); ++i)   { if(m_tex2d[i]) { atomicSafeRelease( m_tex2d[i] ); } }
    for(uint32 i=0; i<_countof(m_tex1d); ++i)   { if(m_tex1d[i]) { atomicSafeRelease( m_tex1d[i] ); } }
    for(uint32 i=0; i<_countof(m_sampler); ++i) { if(m_sampler[i]) { atomicSafeRelease( m_sampler[i] ); } }
    istSafeDelete(m_font);
}


} // namespace atomic
