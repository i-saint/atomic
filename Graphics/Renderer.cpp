#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/World.h"
#include "GPGPU/SPH.cuh"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atomic {


AtomicRenderer* AtomicRenderer::s_inst = NULL;

void AtomicRenderer::initializeInstance()
{
    if(!s_inst) {
        s_inst = istNew(AtomicRenderer) ();
    }
    else {
        istAssert("already initialized");
    }
}

void AtomicRenderer::finalizeInstance()
{
    istSafeDelete(s_inst);
}

AtomicRenderer::AtomicRenderer()
{
    s_inst = this;
    m_va_screenquad = atomicGetVertexArray(VA_SCREEN_QUAD);
    m_sh_out        = atomicGetShader(SH_OUTPUT);

    m_rt_gbuffer    = atomicGetRenderTarget(RT_GBUFFER);
    m_rt_out[0]     = atomicGetRenderTarget(RT_OUTPUT0);
    m_rt_out[1]     = atomicGetRenderTarget(RT_OUTPUT1);

    // 追加の際はデストラクタでの消去処理も忘れずに
    m_renderer_sph              = istNew(PassGBuffer_SPH)();
    m_renderer_bloodstain       = istNew(PassDeferredShading_Bloodstain)();
    m_renderer_lights           = istNew(PassDeferredShading_Lights)();
    m_renderer_microscopic      = istNew(PassPostprocess_Microscopic)();
    m_renderer_fxaa             = istNew(PassPostprocess_FXAA)();
    m_renderer_bloom            = istNew(PassPostprocess_Bloom)();
    m_renderer_fade             = istNew(PassPostprocess_Fade)();
    m_renderer_distance_field   = istNew(PassForwardShading_DistanceField)();
    m_debug_show_gbuffer        = istNew(PassHUD_DebugShowBuffer)();

    m_renderers[PASS_GBUFFER].push_back(m_renderer_sph);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_bloodstain);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_lights);
    m_renderers[PASS_FORWARD].push_back(m_renderer_distance_field);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_fxaa);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_microscopic);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_bloom);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_fade);
    m_renderers[PASS_HUD].push_back(m_debug_show_gbuffer);

    m_stext = istNew(SystemTextRenderer)();

    m_default_viewport = Viewport(ivec2(0), atomicGetWindowSize());
}

AtomicRenderer::~AtomicRenderer()
{
    istSafeDelete(m_stext);
    istSafeDelete(m_renderer_distance_field);
    istSafeDelete(m_renderer_fade);
    istSafeDelete(m_renderer_bloom);
    istSafeDelete(m_renderer_fxaa);
    istSafeDelete(m_renderer_microscopic);
    istSafeDelete(m_renderer_lights);
    istSafeDelete(m_renderer_bloodstain);
    istSafeDelete(m_renderer_sph);
}

void AtomicRenderer::beforeDraw()
{
    for(uint32 i=0; i<_countof(m_renderers); ++i) {
        uint32 size = m_renderers[i].size();
        for(uint32 j=0; j<size; ++j) {
            m_renderers[i][j]->beforeDraw();
        }
    }
    m_stext->beforeDraw();
}


void AtomicRenderer::draw()
{
    PerformanceCounter timer;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);

    {
        PerspectiveCamera *camera   = atomicGetCamera();
        Buffer *ubo_renderstates    = atomicGetUniformBuffer(UBO_RENDER_STATES);
        const uvec2 &wsize          = atomicGetWindowSize();
        camera->updateMatrix();
        m_render_states.ModelViewProjectionMatrix = camera->getModelViewProjectionMatrix();
        m_render_states.CameraPosition  = camera->getPosition();
        m_render_states.ScreenSize      = vec2(atomicGetWindowSize());
        m_render_states.RcpScreenSize   = vec2(1.0f, 1.0f) / m_render_states.ScreenSize;
        m_render_states.AspectRatio     = (float32)wsize.x / (float32)wsize.y;
        m_render_states.RcpAspectRatio  = 1.0f / m_render_states.AspectRatio;
        m_render_states.ScreenTexcoord  = m_render_states.ScreenSize / vec2(m_rt_gbuffer->getColorBuffer(0)->getDesc().size);
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
}

void AtomicRenderer::passShadow()
{
    //glClear(GL_DEPTH_BUFFER_BIT);
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

    if(atomicGetConfig()->enable_multiresolution) {
        m_rt_gbuffer->getColorBuffer(GBUFFER_COLOR)->generateMipmap();
        m_rt_gbuffer->getColorBuffer(GBUFFER_NORMAL)->generateMipmap();
        m_rt_gbuffer->getColorBuffer(GBUFFER_POSITION)->generateMipmap();
        m_rt_gbuffer->getDepthStencilBuffer()->generateMipmap();
    }
}

void AtomicRenderer::passDeferredShading()
{
    RenderTarget *brt = atomicGetBackRenderTarget();
    RenderTarget *rt = atomicGetFrontRenderTarget();

    rt->setDepthStencilBuffer(m_rt_gbuffer->getDepthStencilBuffer());
    rt->bind();
    m_rt_gbuffer->getColorBuffer(GBUFFER_COLOR)->bind(GLSL_COLOR_BUFFER);
    m_rt_gbuffer->getColorBuffer(GBUFFER_NORMAL)->bind(GLSL_NORMAL_BUFFER);
    m_rt_gbuffer->getColorBuffer(GBUFFER_POSITION)->bind(GLSL_POSITION_BUFFER);
    m_rt_gbuffer->getColorBuffer(GBUFFER_GLOW)->bind(GLSL_GLOW_BUFFER);
    brt->getColorBuffer(0)->bind(GLSL_BACK_BUFFER);
    atomicGetTexture2D(TEX2D_RANDOM)->bind(GLSL_RANDOM_BUFFER);

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

    rt->unbind();
    rt->setDepthStencilBuffer(NULL);
}

void AtomicRenderer::passForwardShading()
{
    RenderTarget *rt = atomicGetFrontRenderTarget();
    atomicSwapOutputRenderTarget();

    rt->setDepthStencilBuffer(m_rt_gbuffer->getDepthStencilBuffer());
    rt->bind();
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    uint32 num_renderers = m_renderers[PASS_FORWARD].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_FORWARD][i]->draw();
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    rt->unbind();
    rt->setDepthStencilBuffer(NULL);
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
    atomicGetBackRenderTarget()->getColorBuffer(0)->bind(GLSL_COLOR_BUFFER);
    m_sh_out->bind();
    m_va_screenquad->bind();
    glDrawArrays(GL_QUADS, 0, 4);
    m_sh_out->unbind();

    char buf[64];
    sprintf(buf, "FPS: %.0f", atomicGetRenderingSystem()->getAverageFPS());
    m_stext->addText(ivec2(5, 0), buf);
    sprintf(buf, "Particles: %d", SPHGetStates().fluid_num_particles);
    m_stext->addText(ivec2(5, 20), buf);

    sprintf(buf, "Bloom: [F2]");
    m_stext->addText(ivec2(5, 70), buf);
    {
        const char names[6][32] = {
            "hidden",
            "color",
            "normal",
            "position",
            "glow",
            "all",
        };
        sprintf(buf, "GBuffer: %s [F3/F4]", names[std::abs(atomicGetConfig()->debug_show_gbuffer)%6]);
    }
    m_stext->addText(ivec2(5, 90), buf);
    sprintf(buf, "Lights: %d [F5/F6]", atomicGetConfig()->debug_show_lights);
    m_stext->addText(ivec2(5, 110), buf);
    sprintf(buf, "Pause: [F7]");
    m_stext->addText(ivec2(5, 130), buf);

    m_stext->draw();
}






SystemTextRenderer::SystemTextRenderer()
{
    m_texts.reserve(128);
}

void SystemTextRenderer::beforeDraw()
{
    m_texts.clear();
}

void SystemTextRenderer::draw()
{
    if(!atomicGetConfig()->show_text) { return; }

    for(uint32 i=0; i<m_texts.size(); ++i) {
        const Text &t = m_texts[i];
        atomicGetFont()->draw(t.pos.x, t.pos.y, t.text);
    }
}

void SystemTextRenderer::addText(const ivec2 &pos, const char *text)
{
    Text tmp;
    strncpy(tmp.text, text, _countof(tmp.text));
    tmp.pos = pos;
    m_texts.push_back(tmp);
}



#ifdef __atomic_enable_debug_feature__

void PassHUD_DebugShowBuffer::drawColorBuffer( const DebugShowBufferParams &params )
{
    MapAndWrite(*m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    m_gbuffer->getColorBuffer(GBUFFER_COLOR)->bind(GLSL_COLOR_BUFFER);
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params->getHandle());
    glDrawArrays(GL_QUADS, 0, 4);
    m_sh_rgb->unbind();
}

void PassHUD_DebugShowBuffer::drawNormalBuffer( const DebugShowBufferParams &params )
{
    MapAndWrite(*m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    m_gbuffer->getColorBuffer(GBUFFER_NORMAL)->bind(GLSL_COLOR_BUFFER);
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params->getHandle());
    glDrawArrays(GL_QUADS, 0, 4);
    m_sh_rgb->unbind();
}

void PassHUD_DebugShowBuffer::drawPositionBuffer( const DebugShowBufferParams &params )
{
    MapAndWrite(*m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    m_gbuffer->getColorBuffer(GBUFFER_POSITION)->bind(GLSL_COLOR_BUFFER);
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params->getHandle());
    glDrawArrays(GL_QUADS, 0, 4);
    m_sh_rgb->unbind();
}

void PassHUD_DebugShowBuffer::drawGlowBuffer( const DebugShowBufferParams &params )
{
    MapAndWrite(*m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    m_gbuffer->getColorBuffer(GBUFFER_GLOW)->bind(GLSL_COLOR_BUFFER);
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params->getHandle());
    glDrawArrays(GL_QUADS, 0, 4);
    m_sh_rgb->unbind();
}

PassHUD_DebugShowBuffer::PassHUD_DebugShowBuffer()
{
    m_gbuffer = atomicGetRenderTarget(RT_GBUFFER);
    m_sh_rgb = atomicGetShader(SH_DEBUG_SHOW_RGB);
    m_sh_aaa = atomicGetShader(SH_DEBUG_SHOW_AAA);
    m_ub_params = atomicGetUniformBuffer(UBO_DEBUG_SHOW_BUFFER_PARAMS);
    m_loc_params = m_sh_rgb->getUniformBlockIndex("debug_params");
}
#endif // __atomic_enable_debug_feature__

void PassHUD_DebugShowBuffer::draw()
{
#ifdef __atomic_enable_debug_feature__
    DebugShowBufferParams params;
    params.bottom_left = vec2(-1.0f, -1.0f);
    params.upper_right = vec2( 1.0f,  1.0f);
    params.color_range = vec2( 0.0f, 1.0f);

    m_rt = atomicGetBackRenderTarget();
    m_rt->bind();

    int32 cmd = std::abs(atomicGetConfig()->debug_show_gbuffer) % 6;
    switch(cmd) {
    case 1:
        drawColorBuffer(params);
        break;

    case 2:
        drawNormalBuffer(params);
        break;

    case 3:
        drawPositionBuffer(params);
        break;

    case 4:
        drawGlowBuffer(params);
        break;

    case 5:
        // color
        params.bottom_left = vec2(-1.0f, -1.0f);
        params.upper_right = vec2(-0.5f, -0.5f);
        params.color_range = vec2(0.0f, 1.0f);
        drawColorBuffer(params);

        // normal
        params.bottom_left += vec2(0.5f, 0.0f);
        params.upper_right += vec2(0.5f, 0.0f);
        params.color_range = vec2(0.0f, 1.0f);
        drawNormalBuffer(params);

        // position
        params.bottom_left += vec2(0.5f, 0.0f);
        params.upper_right += vec2(0.5f, 0.0f);
        params.color_range = vec2(0.0f, 1.0f);
        drawPositionBuffer(params);

        // glow
        params.bottom_left += vec2(0.5f, 0.0f);
        params.upper_right += vec2(0.5f, 0.0f);
        params.color_range = vec2(0.0f, 1.0f);
        drawGlowBuffer(params);
        break;
    }

    m_rt->unbind();
#endif // __atomic_enable_debug_feature__
}

} // namespace atomic
