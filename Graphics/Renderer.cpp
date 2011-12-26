#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/World.h"
#include "GPGPU/SPH.cuh"
#include "Renderer.h"
#include "Util.h"

namespace atomic {


AtomicRenderer* AtomicRenderer::s_inst = NULL;

void AtomicRenderer::initializeInstance()
{
    if(!s_inst) {
        s_inst = IST_NEW(AtomicRenderer) ();
    }
    else {
        IST_ASSERT("already initialized");
    }
}

void AtomicRenderer::finalizeInstance()
{
    IST_SAFE_DELETE(s_inst);
}

AtomicRenderer::AtomicRenderer()
{
    m_va_screenquad = atomicGetVertexArray(VA_SCREEN_QUAD);
    m_sh_out        = atomicGetShader(SH_OUTPUT);

    m_rt_gbuffer    = atomicGetRenderTargetGBuffer();
    m_rt_deferred   = atomicGetRenderTargetDeferred();

    m_renderer_sph          = IST_NEW(PassGBuffer_SPH)();
    m_renderer_dir_lights   = IST_NEW(PassDeferredShading_DirectionalLights)();
    m_renderer_point_lights = IST_NEW(PassDeferredShading_PointLights)();
    m_renderer_bloom        = IST_NEW(PassPostprocess_Bloom)();

    m_renderers[PASS_GBUFFER].push_back(m_renderer_sph);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_dir_lights);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_point_lights);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_bloom);

    m_default_viewport.setViewport(0, 0, atomicGetWindowWidth(), atomicGetWindowHeight());
}

AtomicRenderer::~AtomicRenderer()
{
    IST_SAFE_DELETE(m_renderer_bloom);
    IST_SAFE_DELETE(m_renderer_point_lights);
    IST_SAFE_DELETE(m_renderer_dir_lights);
    IST_SAFE_DELETE(m_renderer_sph);
}

void AtomicRenderer::beforeDraw()
{
    for(uint32 i=0; i<_countof(m_renderers); ++i) {
        uint32 size = m_renderers[i].size();
        for(uint32 j=0; j<size; ++j) {
            m_renderers[i][j]->beforeDraw();
        }
    }
}


void AtomicRenderer::draw()
{
    PerformanceCounter timer;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);

    glLoadIdentity();
    {
        PerspectiveCamera *camera = atomicGetCamera();
        UniformBufferObject *ubo_renderstates = atomicGetUniformBufferObject(UBO_RENDER_STATES);
        camera->updateMatrix();
        m_render_states.ModelViewProjectionMatrix = camera->getModelViewProjectionMatrix();
        m_render_states.CameraPosition  = camera->getPosition();
        m_render_states.ScreenSize      = vec2((float32)atomicGetWindowWidth(), (float32)atomicGetWindowHeight());
        m_render_states.RcpScreenSize   = vec2(1.0f, 1.0f) / m_render_states.ScreenSize;
        m_render_states.AspectRatio     = atomicGetWindowAspectRatio();
        m_render_states.RcpAspectRatio  = 1.0f / m_render_states.AspectRatio;
        m_render_states.ScreenTexcoord = vec2(
            m_render_states.ScreenSize.x/float32(m_rt_deferred->getWidth()),
            m_render_states.ScreenSize.y/float32(m_rt_deferred->getHeight()));
        MapAndWrite(*ubo_renderstates, &m_render_states, sizeof(m_render_states));
    }

    passShadow();
    passGBuffer();
    passDeferredShading();
    passForwardShading();
    passPostprocess();
    passHUD();
    passOutput();

    //glFinish();

    timer.count();

    glSwapBuffers();
}

void AtomicRenderer::passShadow()
{
    glClear(GL_DEPTH_BUFFER_BIT);
    glFrontFace(GL_CW);
    glEnable(GL_DEPTH_TEST);

    uint32 num_renderers = m_renderers[PASS_SHADOW_DEPTH].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_SHADOW_DEPTH][i]->draw();
    }

    glDisable(GL_DEPTH_TEST);
    glFrontFace(GL_CCW);
}

void AtomicRenderer::passGBuffer()
{
    const PerspectiveCamera *camera = atomicGetCamera();

    m_rt_gbuffer->bind();
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_ALWAYS, 1, ~0);
    glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);

    uint32 num_renderers = m_renderers[PASS_GBUFFER].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_GBUFFER][i]->draw();
    }

    glDisable(GL_STENCIL_TEST);
    glDisable(GL_DEPTH_TEST);
    m_rt_gbuffer->unbind();
}

void AtomicRenderer::passDeferredShading()
{
    m_rt_deferred->bind();
    m_rt_gbuffer->getColorBuffer(GBUFFER_COLOR)->bind(GLSL_COLOR_BUFFER);
    m_rt_gbuffer->getColorBuffer(GBUFFER_NORMAL)->bind(GLSL_NORMAL_BUFFER);
    m_rt_gbuffer->getColorBuffer(GBUFFER_POSITION)->bind(GLSL_POSITION_BUFFER);
    m_rt_gbuffer->getColorBuffer(GBUFFER_GLOW)->bind(GLSL_GLOW_BUFFER);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);
    glStencilFunc(GL_EQUAL, 1, ~0);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

    uint32 num_renderers = m_renderers[PASS_DEFERRED].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_DEFERRED][i]->draw();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    m_rt_deferred->unbind();
}

void AtomicRenderer::passForwardShading()
{
    uint32 num_renderers = m_renderers[PASS_FORWARD].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_FORWARD][i]->draw();
    }
}

void AtomicRenderer::passPostprocess()
{
    uint32 num_renderers = m_renderers[PASS_POSTPROCESS].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_POSTPROCESS][i]->draw();
    }
}

void AtomicRenderer::passHUD()
{
    uint32 num_renderers = m_renderers[PASS_HUD].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_HUD][i]->draw();
    }
}

void AtomicRenderer::passOutput()
{
    m_rt_deferred->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
    m_sh_out->bind();
    m_va_screenquad->bind();
    glDrawArrays(GL_QUADS, 0, 4);
    m_sh_out->unbind();

    char str_fps[64];
    sprintf(str_fps, "FPS: %.0f", atomicGetApplication()->getAverageFPS());
    atomicGetFont()->draw(0, 0, str_fps);
    sprintf(str_fps, "Particles: %d", SPHGetStates().fluid_num_particles);
    atomicGetFont()->draw(0, 20, str_fps);
}



class UpdateRigidParticle : public AtomicTask
{
private:
    const PSetInstance *m_rinst;
    PSetParticle *m_particles;

public:
    void setup(const PSetInstance &ri, PSetParticle *p)
    {
        m_rinst = &ri;
        m_particles = p;
    }

    void exec()
    {
        const ParticleSet *rc = atomicGetParticleSet(m_rinst->psid);
        uint32 num_particles            = rc->getNumParticles();
        const PSetParticle *particles   = rc->getParticleData();
        simdmat4 t(m_rinst->transform);
        for(uint32 i=0; i<num_particles; ++i) {
            simdvec4 p((vec4&)particles[i].position);
            simdvec4 n((vec4&)particles[i].normal);
            m_particles[i].position = glm::vec4_cast(t * p);
            m_particles[i].normal   = glm::vec4_cast(t * n);
            m_particles[i].diffuse  = m_rinst->diffuse;
            m_particles[i].glow     = m_rinst->glow;
            m_particles[i].flash    = m_rinst->flash;
        }
    }
};


PassGBuffer_SPH::PassGBuffer_SPH()
{
    m_va_cube       = atomicGetVertexArray(VA_FRACTION_CUBE);
    m_sh_fluid      = atomicGetShader(SH_GBUFFER_FLUID);
    m_sh_rigid      = atomicGetShader(SH_GBUFFER_RIGID);
    m_vbo_fluid     = atomicGetVertexBufferObject(VBO_FLUID_PARTICLES);
    m_vbo_rigid     = atomicGetVertexBufferObject(VBO_RIGID_PARTICLES);
}

PassGBuffer_SPH::~PassGBuffer_SPH()
{
    for(uint32 i=0; i<m_tasks.size(); ++i) {
        IST_DELETE(m_tasks[i]);
    }
    m_tasks.clear();
}

void PassGBuffer_SPH::beforeDraw()
{
    m_rinstances.clear();
    m_rparticles.clear();
}

void PassGBuffer_SPH::draw()
{
    // update rigid particle
    uint32 num_rigid_particles = 0;
    uint32 num_rigids = m_rinstances.size();
    {
        resizeTasks(num_rigids);
        for(uint32 i=0; i<num_rigids; ++i) {
            num_rigid_particles += atomicGetParticleSet(m_rinstances[i].psid)->getNumParticles();
        }
        m_rparticles.resize(num_rigid_particles);
        size_t n = 0;
        for(uint32 i=0; i<num_rigids; ++i) {
            static_cast<UpdateRigidParticle*>(m_tasks[i])->setup(m_rinstances[i], &m_rparticles[n]);
            m_tasks[i]->kick();
            n += atomicGetParticleSet(m_rinstances[i].psid)->getNumParticles();
        }
    }
    // copy fluid particles (CUDA -> GL)
    SPHCopyToGL();
    const sphStates& sphs = SPHGetStates();


    // fluid particle
    {
        const uint32 num_particles = sphs.fluid_num_particles;
        const VertexArray::Descriptor descs[] = {
            {GLSL_INSTANCE_PARAM,    VertexArray::TYPE_FLOAT,4,  0, false, 1},
            {GLSL_INSTANCE_POSITION, VertexArray::TYPE_FLOAT,4, 16, false, 1},
            {GLSL_INSTANCE_VELOCITY, VertexArray::TYPE_FLOAT,4, 32, false, 1},
        };
        m_sh_fluid->bind();
        m_va_cube->bind();
        m_va_cube->setAttributes(*m_vbo_fluid, sizeof(sphFluidParticle), descs, _countof(descs));
        glDrawArraysInstanced(GL_QUADS, 0, 24, num_particles);
        m_va_cube->unbind();
        m_sh_fluid->unbind();
    }

    // rigid particle
    {
        for(size_t i=0; i<num_rigids; ++i) {
            m_tasks[i]->join();
        }
        MapAndWrite(*m_vbo_rigid, &m_rparticles[0], sizeof(PSetParticle)*num_rigid_particles);
    }
    {
        const VertexArray::Descriptor descs[] = {
            {GLSL_INSTANCE_POSITION, VertexArray::TYPE_FLOAT,4,  0, false, 1},
            {GLSL_INSTANCE_NORMAL,   VertexArray::TYPE_FLOAT,4, 16, false, 1},
            {GLSL_INSTANCE_COLOR,    VertexArray::TYPE_FLOAT,4, 32, false, 1},
            {GLSL_INSTANCE_GLOW,     VertexArray::TYPE_FLOAT,4, 48, false, 1},
            {GLSL_INSTANCE_PARAM,    VertexArray::TYPE_FLOAT,4, 64, false, 1},
        };
        m_sh_rigid->bind();
        m_va_cube->bind();
        m_va_cube->setAttributes(*m_vbo_rigid, sizeof(PSetParticle), descs, _countof(descs));
        glDrawArraysInstanced(GL_QUADS, 0, 24, num_rigid_particles);
        m_va_cube->unbind();
        m_sh_rigid->unbind();
    }
}

void PassGBuffer_SPH::resizeTasks( uint32 n )
{
    while(m_tasks.size() < n) {
        m_tasks.push_back( IST_NEW(UpdateRigidParticle)() );
    }
}

void PassGBuffer_SPH::addRigidInstance( PSET_RID psid, const mat4 &t, const vec4 &diffuse, const vec4 &glow, const vec4 &flash )
{
    PSetInstance tmp;
    tmp.psid        = psid;
    tmp.transform   = t;
    tmp.diffuse     = diffuse;
    tmp.glow        = glow;
    tmp.flash       = flash;
    m_rinstances.push_back(tmp);
}




PassDeferredShading_DirectionalLights::PassDeferredShading_DirectionalLights()
{
    m_shader        = atomicGetShader(SH_DIRECTIONALLIGHT);
    m_va_quad       = atomicGetVertexArray(VA_SCREEN_QUAD);
    m_vbo_instance  = atomicGetVertexBufferObject(VBO_DIRECTIONALLIGHT_INSTANCES);
    m_instances.reserve(ATOMIC_MAX_DIRECTIONAL_LIGHTS);
}

void PassDeferredShading_DirectionalLights::beforeDraw()
{
    m_instances.clear();
}

void PassDeferredShading_DirectionalLights::draw()
{
    const uint32 num_instances = m_instances.size();
    MapAndWrite(*m_vbo_instance, &m_instances[0], sizeof(light_t)*num_instances);

    const VertexArray::Descriptor descs[] = {
        {GLSL_INSTANCE_DIRECTION,VertexArray::TYPE_FLOAT,4,  0, false, 1},
        {GLSL_INSTANCE_COLOR,    VertexArray::TYPE_FLOAT,4, 16, false, 1},
        {GLSL_INSTANCE_AMBIENT,  VertexArray::TYPE_FLOAT,4, 32, false, 1},
    };
    m_shader->bind();
    m_va_quad->bind();
    m_va_quad->setAttributes(*m_vbo_instance, sizeof(DirectionalLight), descs, _countof(descs));
    glDrawArraysInstanced(GL_QUADS, 0, 4, num_instances);
    m_va_quad->unbind();
    m_shader->unbind();

}

void PassDeferredShading_DirectionalLights::addInstance( const DirectionalLight& v )
{
    m_instances.push_back(v);
}



PassDeferredShading_PointLights::PassDeferredShading_PointLights()
{
    m_shader        = atomicGetShader(SH_POINTLIGHT);
    m_ibo_sphere    = atomicGetIndexBufferObject(IBO_SPHERE);
    m_va_sphere     = atomicGetVertexArray(VA_UNIT_SPHERE);
    m_vbo_instance  = atomicGetVertexBufferObject(VBO_POINTLIGHT_INSTANCES);
    m_instances.reserve(1024);
}

void PassDeferredShading_PointLights::beforeDraw()
{
    m_instances.clear();
}

void PassDeferredShading_PointLights::draw()
{
    const uint32 num_instances = m_instances.size();
    MapAndWrite(*m_vbo_instance, &m_instances[0], sizeof(PointLight)*num_instances);

    const VertexArray::Descriptor descs[] = {
        {GLSL_INSTANCE_POSITION,VertexArray::TYPE_FLOAT,4, 0, false, 1},
        {GLSL_INSTANCE_COLOR,   VertexArray::TYPE_FLOAT,4,16, false, 1},
        {GLSL_INSTANCE_PARAM,   VertexArray::TYPE_FLOAT,4,32, false, 1},
    };

    m_shader->bind();
    m_va_sphere->bind();
    m_va_sphere->setAttributes(*m_vbo_instance, sizeof(PointLight), descs, _countof(descs));
    m_ibo_sphere->bind();
    glDrawElementsInstanced(GL_QUADS, (16-1)*(32)*4, GL_UNSIGNED_INT, 0, num_instances);
    m_ibo_sphere->unbind();
    m_va_sphere->unbind();
    m_shader->unbind();
}


PassPostprocess_Bloom::PassPostprocess_Bloom()
: m_rt_deferred(NULL)
, m_rt_gauss0(NULL)
, m_rt_gauss1(NULL)
, m_va_luminance(NULL)
, m_va_blur(NULL)
, m_va_composite(NULL)
, m_sh_luminance(NULL)
, m_sh_hblur(NULL)
, m_sh_vblur(NULL)
, m_sh_composite(NULL)
, m_ubo_states(NULL)
{
    m_rt_gbuffer    = atomicGetRenderTargetGBuffer();
    m_rt_deferred   = atomicGetRenderTargetDeferred();
    m_rt_gauss0     = atomicGetRenderTargetGauss(0);
    m_rt_gauss1     = atomicGetRenderTargetGauss(1);
    m_va_luminance  = atomicGetVertexArray(VA_BLOOM_LUMINANCE_QUADS);
    m_va_blur       = atomicGetVertexArray(VA_BLOOM_BLUR_QUADS);
    m_va_composite  = atomicGetVertexArray(VA_BLOOM_COMPOSITE_QUAD);
    m_sh_luminance  = atomicGetShader(SH_BLOOM_LUMINANCE);
    m_sh_hblur      = atomicGetShader(SH_BLOOM_HBLUR);
    m_sh_vblur      = atomicGetShader(SH_BLOOM_VBLUR);
    m_sh_composite  = atomicGetShader(SH_BLOOM_COMPOSITE);
    m_ubo_states    = atomicGetUniformBufferObject(UBO_BLOOM_STATES);
}

void PassPostprocess_Bloom::beforeDraw()
{
}

void PassPostprocess_Bloom::draw()
{
    if(!atomicGetConfig()->posteffect_bloom) { return; }

    Viewport vp(0,0, m_rt_gauss0->getWidth(),m_rt_gauss0->getHeight());
    vp.bind();

    // ‹P“x’Šo
    {
        m_sh_luminance->bind();
        m_rt_gauss0->bind();
        m_rt_deferred->getColorBuffer(GBUFFER_COLOR)->bind(GLSL_COLOR_BUFFER);
        m_rt_gbuffer->getColorBuffer(GBUFFER_GLOW)->bind(GLSL_GLOW_BUFFER);
        m_va_luminance->bind();
        glDrawArrays(GL_QUADS, 0, 16);
        m_rt_gauss0->unbind();
        m_sh_luminance->unbind();
    }

    // ‰¡ƒuƒ‰[
    {
        m_sh_hblur->bind();
        m_rt_gauss1->bind();
        m_rt_gauss0->getColorBuffer(GBUFFER_COLOR)->bind(GLSL_COLOR_BUFFER);
        m_va_blur->bind();
        glDrawArrays(GL_QUADS, 0, 16);
        m_rt_gauss1->unbind();
        m_sh_hblur->unbind();
    }

    // cƒuƒ‰[
    {
        m_sh_vblur->bind();
        m_rt_gauss0->bind();
        m_rt_gauss1->getColorBuffer(GBUFFER_COLOR)->bind(GLSL_COLOR_BUFFER);
        m_va_blur->bind();
        glDrawArrays(GL_QUADS, 0, 16);
        m_rt_gauss0->unbind();
        m_sh_vblur->unbind();
    }

    // ‰ÁŽZ
    atomicGetDefaultViewport()->bind();
    {
        m_sh_composite->bind();
        m_rt_deferred->bind();
        m_rt_gauss0->getColorBuffer(GBUFFER_COLOR)->bind(GLSL_COLOR_BUFFER);
        m_va_composite->bind();
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDrawArrays(GL_QUADS, 0, 4);
        glDisable(GL_BLEND);
        m_rt_gauss0->getColorBuffer(0)->unbind();
        m_rt_deferred->unbind();
        m_sh_composite->unbind();
    }
}


} // namespace atomic
