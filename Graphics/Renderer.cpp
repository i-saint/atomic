#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/World.h"
#include "GPGPU/SPH.cuh"
#include "Renderer.h"

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
    m_sh_out        = atomicGetShader(SH_OUTPUT);

    m_rt_gbuffer    = atomicGetRenderTargetGBuffer();
    m_rt_deferred   = atomicGetRenderTargetDeferred();

    m_renderer_cube = IST_NEW(PassGBuffer_Cube) ();
    m_renderer_sphere_light = IST_NEW(PassDeferred_PointLight) ();
    m_renderer_bloom = IST_NEW(PassPostprocess_Bloom) ();

    m_renderers[PASS_GBUFFER].push_back(m_renderer_cube);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_sphere_light);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_bloom);

    m_default_viewport.setViewport(0, 0, atomicGetWindowWidth(), atomicGetWindowHeight());
}

AtomicRenderer::~AtomicRenderer()
{
    IST_SAFE_DELETE(m_renderer_bloom);
    IST_SAFE_DELETE(m_renderer_sphere_light);
    IST_SAFE_DELETE(m_renderer_cube);
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
        PerformanceCounter counter;
        SPHCopyInstancePositions();
        //IST_PRINT("SPHCopyInstancePositions() took %f ms.\n", counter.getElapsedMillisecond());
    }
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

    pass_Shadow();
    pass_GBuffer();
    pass_Deferred();
    pass_Forward();
    pass_Postprocess();
    pass_Output();
    pass_UI();

    //glFinish();

    timer.count();

    glSwapBuffers();
}

void AtomicRenderer::pass_Shadow()
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

void AtomicRenderer::pass_GBuffer()
{
    const PerspectiveCamera *camera = atomicGetCamera();

    m_rt_gbuffer->bind();
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    uint32 num_renderers = m_renderers[PASS_GBUFFER].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_GBUFFER][i]->draw();
    }

    glDisable(GL_DEPTH_TEST);
    m_rt_gbuffer->unbind();
}

void AtomicRenderer::pass_Deferred()
{
    m_rt_deferred->bind();
    m_rt_gbuffer->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
    m_rt_gbuffer->getColorBuffer(1)->bind(GLSL_NORMAL_BUFFER);
    m_rt_gbuffer->getColorBuffer(2)->bind(GLSL_POSITION_BUFFER);
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);

    uint32 num_renderers = m_renderers[PASS_DEFERRED].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_DEFERRED][i]->draw();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    m_rt_deferred->unbind();
}

void AtomicRenderer::pass_Forward()
{
    uint32 num_renderers = m_renderers[PASS_FORWARD].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_FORWARD][i]->draw();
    }
}

void AtomicRenderer::pass_Postprocess()
{
    uint32 num_renderers = m_renderers[PASS_POSTPROCESS].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_POSTPROCESS][i]->draw();
    }
}

void AtomicRenderer::pass_UI()
{
    uint32 num_renderers = m_renderers[PASS_UI].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_UI][i]->draw();
    }

    char str_fps[64];
    sprintf(str_fps, "FPS: %.0f", atomicGetApplication()->getAverageFPS());
    atomicGetFont()->draw(0, 0, str_fps);
}

void AtomicRenderer::pass_Output()
{
    m_rt_deferred->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
    m_sh_out->bind();
    glDrawArrays(GL_TRIANGLES, 0, 6);
    m_sh_out->unbind();
}



PassGBuffer_Cube::PassGBuffer_Cube()
{
    m_sh_gbuffer        = atomicGetShader(SH_GBUFFER);
    m_model             = atomicGetModelData(MODEL_CUBE_FRACTION);
    m_vbo_fraction_pos  = atomicGetVertexBufferObject(VBO_FRACTION_POS);
}

void PassGBuffer_Cube::beforeDraw()
{
    m_vfx.clear();
}

void PassGBuffer_Cube::draw()
{
    //const uint32 num_fractions = m_fraction.pos.size();
    //m_vbo_instance_pos->allocate(sizeof(XMVECTOR)*num_fractions, VertexBufferObject::USAGE_STREAM, &m_fraction.pos[0]);

    const uint32 num_fractions = SPH_MAX_PARTICLE_NUM;

    m_sh_gbuffer->bind();
    m_model->setInstanceData(GLSL_INSTANCE_POSITION, 4, *m_vbo_fraction_pos);
    m_model->drawInstanced(num_fractions);
    m_sh_gbuffer->unbind();
}



PassDeferred_PointLight::PassDeferred_PointLight()
{
    m_shader = atomicGetShader(SH_POINTLIGHT);
    m_model = atomicGetModelData(MODEL_SPHERE_LIGHT);
    m_vbo_instance_pos = atomicGetVertexBufferObject(VBO_POINTLIGHT_POS);
    m_instance_pos.reserve(1024);
}

void PassDeferred_PointLight::beforeDraw()
{
    m_instance_pos.clear();
}

void PassDeferred_PointLight::draw()
{
    //const uint32 num_instances = m_instance_pos.size();
    //m_vbo_instance_pos->allocate(sizeof(XMVECTOR)*num_instances, VertexBufferObject::USAGE_STREAM, &m_instance_pos[0]);

    const uint32 num_instances = SPH_MAX_LIGHT_NUM;

    m_shader->bind();
    m_model->setInstanceData(GLSL_INSTANCE_POSITION, 4, *m_vbo_instance_pos);
    m_model->drawInstanced(num_instances);
    m_shader->unbind();
}


PassPostprocess_Bloom::PassPostprocess_Bloom()
: m_rt_deferred(NULL)
, m_rt_gauss0(NULL)
, m_rt_gauss1(NULL)
, m_sh_luminance(NULL)
, m_sh_hblur(NULL)
, m_sh_vblur(NULL)
, m_sh_composite(NULL)
, m_ubo_states(NULL)
, m_loc_state(0)
{
    m_rt_deferred   = atomicGetRenderTargetDeferred();
    m_rt_gauss0     = atomicGetRenderTargetGauss(0);
    m_rt_gauss1     = atomicGetRenderTargetGauss(1);
    m_sh_luminance  = atomicGetShader(SH_BLOOM_LUMINANCE);
    m_sh_hblur      = atomicGetShader(SH_BLOOM_HBLUR);
    m_sh_vblur      = atomicGetShader(SH_BLOOM_VBLUR);
    m_sh_composite  = atomicGetShader(SH_BLOOM_COMPOSITE);
    m_ubo_states    = atomicGetUniformBufferObject(UBO_BLOOM_STATES);
    m_loc_state     = m_sh_luminance->getUniformBlockIndex("bloom_states");
}

void PassPostprocess_Bloom::beforeDraw()
{
}

void PassPostprocess_Bloom::draw()
{
    const float32 aspect_ratio = atomicGetWindowAspectRatio();
    const float32 rcp_aspect_ratio = 1.0f/aspect_ratio;
    const Viewport viewports[] = {
        Viewport(  0,0, 256,256),
        Viewport(256,0, 128,128),
        Viewport(384,0,  64, 64),
        Viewport(448,0,  32, 32),
    };
    const vec2 texcoord_pos[] = {
        vec2( 0.0f, 0.0f),
        vec2( 0.5f, 0.0f),
        vec2(0.75f, 0.0f),
        vec2(0.875f, 0.0f),
    };
    const vec2 texcoord_size[] = {
        vec2(  0.5f,  1.0f),
        vec2( 0.25f,  0.5f),
        vec2(0.125f, 0.25f),
        vec2(0.0625f, 0.125f),
    };
    const vec2 screen_size = vec2((float32)m_rt_gauss0->getWidth(), (float32)m_rt_gauss0->getHeight());
    const vec2 rcp_screen_size = vec2(1.0f, 1.0f) / screen_size;
    const vec2 half_pixel = rcp_screen_size*0.5f;


    // ‹P“x’Šo
    {
        m_sh_luminance->bind();
        m_rt_gauss0->bind();
        m_rt_deferred->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 1);
        m_rt_gauss0->unbind();
        m_sh_luminance->unbind();
    }

    //// ‰¡ƒuƒ‰[
    //{
    //    int loc_tmin = m_sh_hblur->getUniformLocation("u_TexcoordMin");
    //    int loc_tmax = m_sh_hblur->getUniformLocation("u_TexcoordMax");
    //    m_sh_hblur->bind();
    //    m_rt_gauss1->bind();
    //    m_rt_gauss0->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
    //    for(uint32 i=0; i<1; ++i) {
    //        viewports[i].bind();
    //        vec2 tmin = texcoord_pos[i] + half_pixel;
    //        vec2 tmax = texcoord_pos[i]+texcoord_size[i] - half_pixel;
    //        tmax.y *= rcp_aspect_ratio;
    //        m_sh_hblur->setUniform2f(loc_tmin, tmin);
    //        m_sh_hblur->setUniform2f(loc_tmax, tmax);
    //        glDrawArrays(GL_TRIANGLES, 0, 6);
    //    }
    //    m_rt_gauss0->getColorBuffer(0)->unbind();
    //    m_rt_gauss1->unbind();
    //    m_sh_hblur->unbind();
    //}

    //// cƒuƒ‰[
    //{
    //    int loc_tmin = m_sh_hblur->getUniformLocation("u_TexcoordMin");
    //    int loc_tmax = m_sh_hblur->getUniformLocation("u_TexcoordMax");
    //    m_sh_vblur->bind();
    //    m_rt_gauss0->bind();
    //    m_rt_gauss1->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
    //    for(uint32 i=0; i<_countof(viewports); ++i) {
    //        viewports[i].bind();
    //        vec2 tmin = texcoord_pos[i] + half_pixel;
    //        vec2 tmax = texcoord_pos[i]+texcoord_size[i] - half_pixel;
    //        tmax.y *= rcp_aspect_ratio;
    //        m_sh_vblur->setUniform2f(loc_tmin, tmin);
    //        m_sh_vblur->setUniform2f(loc_tmax, tmax);
    //        DrawScreen(texcoord_pos[i], texcoord_pos[i]+texcoord_size[i]);
    //    }
    //    m_rt_gauss1->getColorBuffer(0)->unbind();
    //    m_rt_gauss0->unbind();
    //    m_sh_vblur->unbind();
    //}

    //// ‰ÁŽZ
    //atomicGetDefaultViewport()->bind();
    //{
    //    m_sh_bloom->switchToCompositePass();
    //    m_rt_deferred->bind();
    //    m_rt_gauss0->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
    //    m_sh_bloom->setColorBuffer(GLSL_COLOR_BUFFER);
    //    glEnable(GL_BLEND);
    //    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    //    for(uint32 i=0; i<_countof(viewports); ++i) {
    //        vec2 tmin = texcoord_pos[i] + half_pixel;
    //        vec2 tmax = texcoord_pos[i]+texcoord_size[i] - half_pixel;
    //        tmax.y *= rcp_aspect_ratio;
    //        DrawScreen(tmin, tmax);
    //    }
    //    glDisable(GL_BLEND);
    //    m_rt_gauss0->getColorBuffer(0)->unbind();
    //    m_rt_deferred->unbind();
    //}
    //m_sh_bloom->unbind();

    atomicGetDefaultViewport()->bind();
}

} // namespace atomic
