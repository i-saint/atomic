#include "stdafx.h"
#include "ist/iui.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atomic {


AtomicRenderer* AtomicRenderer::s_inst = NULL;

void AtomicRenderer::initializeInstance()
{
    istAssert(s_inst==NULL);
    s_inst = istNew(AtomicRenderer) ();
}

void AtomicRenderer::finalizeInstance()
{
    istSafeDelete(s_inst);
}

AtomicRenderer::AtomicRenderer()
{
    istMemset(&m_rstates3d, 0, sizeof(m_rstates3d));

    s_inst = this;
    m_va_screenquad = atomicGetVertexArray(VA_SCREEN_QUAD);
    m_sh_out        = atomicGetShader(SH_OUTPUT);

    m_rt_gbuffer    = atomicGetRenderTarget(RT_GBUFFER);
    m_rt_out[0]     = atomicGetRenderTarget(RT_OUTPUT0);
    m_rt_out[1]     = atomicGetRenderTarget(RT_OUTPUT1);

    // 追加の際はデストラクタでの消去処理も忘れずに
    m_renderer_fluid            = istNew(PassGBuffer_Fluid)();
    m_renderer_particle         = istNew(PassGBuffer_Particle)();
    m_renderer_bg               = istNew(PassGBuffer_BG);
    m_renderer_bloodstain       = istNew(PassDeferredShading_Bloodstain)();
    m_renderer_lights           = istNew(PassDeferredShading_Lights)();
    m_renderer_microscopic      = istNew(PassPostprocess_Microscopic)();
    m_renderer_fxaa             = istNew(PassPostprocess_FXAA)();
    m_renderer_bloom            = istNew(PassPostprocess_Bloom)();
    m_renderer_fade             = istNew(PassPostprocess_Fade)();
    m_renderer_distance_field   = istNew(PassForwardShading_DistanceField)();
#ifdef atomic_enable_gbuffer_viewer
    m_debug_show_gbuffer        = istNew(PassHUD_DebugShowBuffer)();
#endif // atomic_enable_gbuffer_viewer

    m_stext = istNew(SystemTextRenderer)();

    m_renderers[PASS_GBUFFER].push_back(m_renderer_fluid);
    m_renderers[PASS_GBUFFER].push_back(m_renderer_particle);
    m_renderers[PASS_GBUFFER].push_back(m_renderer_bg);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_bloodstain);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_lights);
    m_renderers[PASS_FORWARD].push_back(m_renderer_distance_field);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_fxaa);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_microscopic);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_bloom);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_fade);
#ifdef atomic_enable_gbuffer_viewer
    m_renderers[PASS_HUD].push_back(m_debug_show_gbuffer);
#endif // atomic_enable_gbuffer_viewer

    m_default_viewport = Viewport(ivec2(0), atomicGetWindowSize());
}

AtomicRenderer::~AtomicRenderer()
{
    istSafeDelete(m_stext);
#ifdef atomic_enable_gbuffer_viewer
    istSafeDelete(m_debug_show_gbuffer);
#endif // atomic_enable_gbuffer_viewer
    istSafeDelete(m_renderer_distance_field);
    istSafeDelete(m_renderer_fade);
    istSafeDelete(m_renderer_bloom);
    istSafeDelete(m_renderer_fxaa);
    istSafeDelete(m_renderer_microscopic);
    istSafeDelete(m_renderer_lights);
    istSafeDelete(m_renderer_bloodstain);
    istSafeDelete(m_renderer_bg);
    istSafeDelete(m_renderer_particle);
    istSafeDelete(m_renderer_fluid);
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
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);

    {
        PerspectiveCamera *camera   = atomicGetGameCamera();
        Buffer *ubo_rs              = atomicGetUniformBuffer(UBO_RENDERSTATES_3D);
        const uvec2 &wsize          = atomicGetWindowSize();
        m_rstates3d.ModelViewProjectionMatrix = camera->getViewProjectionMatrix();
        m_rstates3d.CameraPosition  = camera->getPosition();
        m_rstates3d.CameraDirection = camera->getDirection();
        m_rstates3d.ScreenSize      = vec2(atomicGetWindowSize());
        m_rstates3d.RcpScreenSize   = vec2(1.0f, 1.0f) / m_rstates3d.ScreenSize;
        m_rstates3d.AspectRatio     = (float32)wsize.x / (float32)wsize.y;
        m_rstates3d.RcpAspectRatio  = 1.0f / m_rstates3d.AspectRatio;
        m_rstates3d.ScreenTexcoord  = m_rstates3d.ScreenSize / vec2(m_rt_gbuffer->getColorBuffer(0)->getDesc().size);
        m_rstates3d.Color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
        m_rstates3d.Frame = (float32)atomicGetFrame();
        MapAndWrite(dc, ubo_rs, &m_rstates3d, sizeof(m_rstates3d));

        ubo_rs              = atomicGetUniformBuffer(UBO_RENDERSTATES_BG);
        m_rstatesBG = m_rstates3d;
        MapAndWrite(dc, ubo_rs, &m_rstatesBG, sizeof(m_rstatesBG));
    }
    {
        Buffer *ubo_rs      = atomicGetUniformBuffer(UBO_RENDERSTATES_2D);
        const vec2 &wsize   = vec2(atomicGetWindowSize());
        m_rstates2d         = m_rstates3d;
        m_rstates2d.ModelViewProjectionMatrix = glm::ortho(0.0f, wsize.x, wsize.y, 0.0f);
        MapAndWrite(dc, ubo_rs, &m_rstates2d, sizeof(m_rstates2d));
    }
    {
        Sampler *smp_gb = atomicGetSampler(SAMPLER_GBUFFER);
        Sampler *smp_tex = atomicGetSampler(SAMPLER_TEXTURE_DEFAULT);
        dc->setSampler(GLSL_COLOR_BUFFER, smp_tex);
        dc->setSampler(GLSL_NORMAL_BUFFER, smp_gb);
        dc->setSampler(GLSL_POSITION_BUFFER, smp_gb);
        dc->setSampler(GLSL_GLOW_BUFFER, smp_tex);
        dc->setSampler(GLSL_BACK_BUFFER, smp_tex);
        dc->setSampler(GLSL_RANDOM_BUFFER, smp_tex);
        dc->setSampler(GLSL_PARAM_BUFFER, smp_tex);
        dc->setBlendState(atomicGetBlendState(BS_NO_BLEND));
    }

    passShadow();
    passGBuffer();
    passDeferredShading();
    passForwardShading();
    passPostprocess();
    passHUD();
    passOutput();

    //glFinish();
}

void AtomicRenderer::passShadow()
{
    glFrontFace(GL_CW);

    uint32 num_renderers = m_renderers[PASS_SHADOW_DEPTH].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_SHADOW_DEPTH][i]->draw();
    }

    glFrontFace(GL_CCW);
}

void AtomicRenderer::passGBuffer()
{
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    const PerspectiveCamera *camera = atomicGetGameCamera();

    dc->clearColor(m_rt_gbuffer, vec4(0.0f,0.0f,0.0f,1.0f));
    dc->clearDepthStencil(m_rt_gbuffer, 1.0f, 0);
    dc->setRenderTarget(m_rt_gbuffer);

    uint32 num_renderers = m_renderers[PASS_GBUFFER].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_GBUFFER][i]->draw();
    }

    dc->setDepthStencilState(atomicGetDepthStencilState(DS_NO_DEPTH_NO_STENCIL));
    dc->setRenderTarget(NULL);

    if(atomicGetConfig()->light_multiresolution) {
        dc->generateMips(m_rt_gbuffer->getColorBuffer(GBUFFER_COLOR));
        dc->generateMips(m_rt_gbuffer->getColorBuffer(GBUFFER_COLOR));
        dc->generateMips(m_rt_gbuffer->getColorBuffer(GBUFFER_NORMAL));
        dc->generateMips(m_rt_gbuffer->getColorBuffer(GBUFFER_POSITION));
    }
}

void AtomicRenderer::passDeferredShading()
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    RenderTarget *brt = atomicGetBackRenderTarget();
    RenderTarget *rt = atomicGetFrontRenderTarget();

    rt->setDepthStencilBuffer(m_rt_gbuffer->getDepthStencilBuffer());
    dc->clearColor(rt, vec4(0.0f, 0.0f, 0.0f, 1.0f));
    dc->setRenderTarget(rt);
    dc->setTexture(GLSL_COLOR_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_COLOR));
    dc->setTexture(GLSL_NORMAL_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_NORMAL));
    dc->setTexture(GLSL_POSITION_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_POSITION));
    dc->setTexture(GLSL_GLOW_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_GLOW));
    dc->setTexture(GLSL_BACK_BUFFER, brt->getColorBuffer(0));
    dc->setTexture(GLSL_RANDOM_BUFFER, atomicGetTexture2D(TEX2D_RANDOM));
    dc->setBlendState(atomicGetBlendState(BS_BLEND_ADD));
    dc->setDepthStencilState(atomicGetDepthStencilState(DS_LIGHTING_FRONT));

    uint32 num_renderers = m_renderers[PASS_DEFERRED].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_DEFERRED][i]->draw();
    }
    dc->setDepthStencilState(atomicGetDepthStencilState(DS_NO_DEPTH_NO_STENCIL));
    dc->setBlendState(atomicGetBlendState(BS_NO_BLEND));

    dc->setTexture(GLSL_COLOR_BUFFER, NULL);
    dc->setTexture(GLSL_NORMAL_BUFFER, NULL);
    dc->setTexture(GLSL_POSITION_BUFFER, NULL);
    dc->setTexture(GLSL_GLOW_BUFFER, NULL);
    rt->setDepthStencilBuffer(NULL);
}

void AtomicRenderer::passForwardShading()
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    RenderTarget *rt = atomicGetFrontRenderTarget();
    atomicSwapOutputRenderTarget();

    rt->setDepthStencilBuffer(m_rt_gbuffer->getDepthStencilBuffer());
    dc->setRenderTarget(rt);

    uint32 num_renderers = m_renderers[PASS_FORWARD].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_FORWARD][i]->draw();
    }

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
}

void AtomicRenderer::passOutput()
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    m_sh_out->assign(dc);
    dc->setRenderTarget(NULL);
    dc->setTexture(GLSL_COLOR_BUFFER, atomicGetBackRenderTarget()->getColorBuffer(0));
    dc->setVertexArray(m_va_screenquad);
    dc->draw(I3D_QUADS, 0, 4);

    {
        uint32 num_renderers = m_renderers[PASS_HUD].size();
        for(uint32 i=0; i<num_renderers; ++i) {
            m_renderers[PASS_HUD][i]->draw();
        }
    }

    // texts 

    char buf[64];
    istSPrintf(buf, "FPS: %u", atomicGetRenderingSystem()->getAverageFPS());
    m_stext->addText(vec2(5.0f, 5.0f), buf);
    istSPrintf(buf, "Particles: %d", atomicGetSPHManager()->getNumParticles());
    m_stext->addText(vec2(5.0f, 25.0f), buf);

    //istsprintf(buf, "Bloom: [F2]");
    //m_stext->addText(vec2(5.0f, 45.0f), buf);
    //{
    //    const char names[6][32] = {
    //        "hidden",
    //        "color",
    //        "normal",
    //        "position",
    //        "glow",
    //        "all",
    //    };
    //    istsprintf(buf, "GBuffer: %s [F3/F4]", names[std::abs(atomicGetConfig()->debug_show_gbuffer)%6]);
    //}
    //m_stext->addText(vec2(5.0f, 110.0f), buf);
    //istsprintf(buf, "Lights: %d [F5/F6]", atomicGetConfig()->debug_show_lights);
    //m_stext->addText(vec2(5.0f, 130.0f), buf);
    //istsprintf(buf, "Pause: [F7]");
    //m_stext->addText(vec2(5.0f, 150.0f), buf);
    //istsprintf(buf, "Toggle Multiresolution: [F8]");
    //m_stext->addText(vec2(5.0f, 170.0f), buf);
    //istsprintf(buf, "Show Multiresolution Level: [F9]");
    //m_stext->addText(vec2(5.0f, 190.0f), buf);

    //istsprintf(buf, "Multiresolution Threshold: %.3f ([8]<- [9]->)", atomicGetLights()->getMultiresolutionParams().Threshold.x);
    //m_stext->addText(vec2(5.0f, 210.0f), buf);

    dc->setBlendState(atomicGetBlendState(BS_BLEND_ALPHA));

    iuiDraw();
    iuiDrawFlush();
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
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();

    {
        const vec2 &wsize   = vec2(atomicGetWindowSize());
        auto *font = atomicGetFontRenderer();
        font->setScreen(0.0f, wsize.x, wsize.y, 0.0f);
        font->setSize(18.0f);
        font->setMonospace(true);
        font->setSpacing(0.75f);
        font->setColor(vec4(1.0f));
    }
    for(uint32 i=0; i<m_texts.size(); ++i) {
        const Text &t = m_texts[i];
        atomicGetFontRenderer()->addText(t.pos, t.text, wcsnlen(t.text, _countof(t.text)));
    }
    atomicGetFontRenderer()->flush(atomicGetGLDeviceContext());
}

void SystemTextRenderer::addText(const vec2 &pos, const char *text)
{
    Text tmp;
    mbstowcs(tmp.text, text, _countof(tmp.text));
    tmp.pos = pos;
    m_texts.push_back(tmp);
}

void SystemTextRenderer::addText( const vec2 &pos, const wchar_t *text )
{
    Text tmp;
    wcsncpy(tmp.text, text, _countof(tmp.text));
    tmp.pos = pos;
    m_texts.push_back(tmp);
}



#ifdef atomic_enable_gbuffer_viewer

void PassHUD_DebugShowBuffer::drawColorBuffer( const DebugShowBufferParams &params )
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    MapAndWrite(dc, m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    dc->setTexture(GLSL_COLOR_BUFFER, m_gbuffer->getColorBuffer(GBUFFER_COLOR));
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params);
    dc->draw(I3D_QUADS, 0, 4);
    m_sh_rgb->unbind();
}

void PassHUD_DebugShowBuffer::drawNormalBuffer( const DebugShowBufferParams &params )
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    MapAndWrite(dc, m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    dc->setTexture(GLSL_COLOR_BUFFER, m_gbuffer->getColorBuffer(GBUFFER_NORMAL));
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params);
    dc->draw(I3D_QUADS, 0, 4);
    m_sh_rgb->unbind();
}

void PassHUD_DebugShowBuffer::drawPositionBuffer( const DebugShowBufferParams &params )
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    MapAndWrite(dc, m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    dc->setTexture(GLSL_COLOR_BUFFER, m_gbuffer->getColorBuffer(GBUFFER_POSITION));
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params);
    dc->draw(I3D_QUADS, 0, 4);
    m_sh_rgb->unbind();
}

void PassHUD_DebugShowBuffer::drawGlowBuffer( const DebugShowBufferParams &params )
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    MapAndWrite(dc, m_ub_params, &params, sizeof(params));

    m_sh_rgb->bind();
    dc->setTexture(GLSL_COLOR_BUFFER, m_gbuffer->getColorBuffer(GBUFFER_GLOW));
    m_sh_rgb->setUniformBlock(m_loc_params, GLSL_DEBUG_BUFFER_BINDING, m_ub_params);
    dc->draw(I3D_QUADS, 0, 4);
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

void PassHUD_DebugShowBuffer::draw()
{
    DebugShowBufferParams params;
    params.BottomLeft = vec2(-1.0f, -1.0f);
    params.UpperRight = vec2( 1.0f,  1.0f);
    params.ColorRange = vec2( 0.0f, 1.0f);

    //m_rt = atomicGetBackRenderTarget();
    //m_rt->bind();

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
        params.BottomLeft = vec2(-1.0f, -1.0f);
        params.UpperRight = vec2(-0.5f, -0.5f);
        params.ColorRange = vec2(0.0f, 1.0f);
        drawColorBuffer(params);

        // normal
        params.BottomLeft += vec2(0.5f, 0.0f);
        params.UpperRight += vec2(0.5f, 0.0f);
        params.ColorRange = vec2(0.0f, 1.0f);
        drawNormalBuffer(params);

        // position
        params.BottomLeft += vec2(0.5f, 0.0f);
        params.UpperRight += vec2(0.5f, 0.0f);
        params.ColorRange = vec2(0.0f, 1.0f);
        drawPositionBuffer(params);

        // glow
        params.BottomLeft += vec2(0.5f, 0.0f);
        params.UpperRight += vec2(0.5f, 0.0f);
        params.ColorRange = vec2(0.0f, 1.0f);
        drawGlowBuffer(params);
        break;
    }

    //m_rt->unbind();
}
#endif // atomic_enable_gbuffer_viewer

} // namespace atomic
