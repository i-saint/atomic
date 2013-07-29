#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atm {


PassPostprocess_Microscopic::PassPostprocess_Microscopic()
{
    m_rt_gbuffer    = atmGetRenderTarget(RT_GBUFFER);
    m_va_quad       = atmGetVertexArray(VA_SCREEN_QUAD);
    m_sh            = atmGetShader(SH_MICROSCOPIC);
}

void PassPostprocess_Microscopic::beforeDraw()
{
}

void PassPostprocess_Microscopic::draw()
{
    if(!atmGetConfig()->posteffect_microscopic) { return; }

    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    RenderTarget *brt = atmGetBackRenderTarget();
    RenderTarget *rt = atmGetFrontRenderTarget();
    atmSwapOutputRenderTarget();

    m_sh->assign(dc);
    dc->setRenderTarget(rt);
    dc->setTexture(GLSL_COLOR_BUFFER, brt->getColorBuffer(0));
    dc->setTexture(GLSL_NORMAL_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_NORMAL));
    dc->setTexture(GLSL_POSITION_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_POSITION));
    dc->setTexture(GLSL_GLOW_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_GLOW));
    dc->setVertexArray(m_va_quad);
    dc->draw(I3D_QUADS, 0, 4);
}


PassPostprocess_FXAA::PassPostprocess_FXAA()
{
    m_sh_FXAA_luma  = atmGetShader(SH_FXAA_LUMA);
    m_sh_FXAA       = atmGetShader(SH_FXAA);
    m_va_quad       = atmGetVertexArray(VA_SCREEN_QUAD);
    m_loc_fxaa_param = m_sh_FXAA->getUniformBlockIndex("fxaa_params");
}

void PassPostprocess_FXAA::beforeDraw()
{
}

void PassPostprocess_FXAA::draw()
{
    if(!atmGetConfig()->posteffect_antialias) { return; }

    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    Buffer *ubo_fxaa                        = atmGetUniformBuffer(UBO_FXAA_PARAMS);
    m_fxaaparams.fxaaQualityRcpFrame        = vec2(1.0f, 1.0f) / vec2(atmGetWindowSize());
    m_fxaaparams.fxaaQualitySubpix          = 0.75f;
    m_fxaaparams.fxaaQualityEdgeThreshold   = 0.166f;
    m_fxaaparams.fxaaQualityEdgeThresholdMin= 0.0833f;
    MapAndWrite(dc, ubo_fxaa, &m_fxaaparams, sizeof(m_fxaaparams));

    // 輝度抽出
    {
        RenderTarget *brt = atmGetBackRenderTarget();
        RenderTarget *rt = atmGetFrontRenderTarget();
        atmSwapOutputRenderTarget();

        dc->setRenderTarget(rt);
        m_sh_FXAA_luma->assign(dc);
        dc->setTexture(GLSL_COLOR_BUFFER, brt->getColorBuffer(GBUFFER_COLOR));
        dc->setVertexArray(m_va_quad);
        dc->draw(I3D_QUADS, 0, 4);
        dc->setShader(nullptr);
        dc->setRenderTarget(nullptr);
    }
    // FXAA
    {
        RenderTarget *brt = atmGetBackRenderTarget();
        RenderTarget *rt = atmGetFrontRenderTarget();
        atmSwapOutputRenderTarget();

        dc->setRenderTarget(rt);
        m_sh_FXAA->assign(dc);
        m_sh_FXAA->setUniformBlock(m_loc_fxaa_param, GLSL_FXAA_BINDING, ubo_fxaa);
        dc->setTexture(GLSL_COLOR_BUFFER, brt->getColorBuffer(0));
        dc->setVertexArray(m_va_quad);
        dc->draw(I3D_QUADS, 0, 4);
        dc->setShader(nullptr);
        dc->setRenderTarget(nullptr);
    }
}



PassPostprocess_Bloom::PassPostprocess_Bloom()
: m_rt_gauss0(nullptr)
, m_rt_gauss1(nullptr)
, m_va_luminance(nullptr)
, m_va_blur(nullptr)
, m_va_composite(nullptr)
, m_sh_luminance(nullptr)
, m_sh_hblur(nullptr)
, m_sh_vblur(nullptr)
, m_sh_composite(nullptr)
, m_ubo_states(nullptr)
{
    m_rt_gbuffer    = atmGetRenderTarget(RT_GBUFFER);
    m_rt_gauss0     = atmGetRenderTarget(RT_GAUSS0);
    m_rt_gauss1     = atmGetRenderTarget(RT_GAUSS1);
    m_va_luminance  = atmGetVertexArray(VA_BLOOM_LUMINANCE_QUADS);
    m_va_blur       = atmGetVertexArray(VA_BLOOM_BLUR_QUADS);
    m_va_composite  = atmGetVertexArray(VA_BLOOM_COMPOSITE_QUAD);
    m_sh_luminance  = atmGetShader(SH_BLOOM_LUMINANCE);
    m_sh_hblur      = atmGetShader(SH_BLOOM_HBLUR);
    m_sh_vblur      = atmGetShader(SH_BLOOM_VBLUR);
    m_sh_composite  = atmGetShader(SH_BLOOM_COMPOSITE);
    m_ubo_states    = atmGetUniformBuffer(UBO_BLOOM_PARAMS);
}

void PassPostprocess_Bloom::beforeDraw()
{
}

void PassPostprocess_Bloom::draw()
{
    if(!atmGetConfig()->posteffect_bloom) { return; }


    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    Viewport vp(ivec2(), m_rt_gauss0->getColorBuffer(0)->getDesc().size);
    dc->setViewport(vp);

    // 輝度抽出
    {
        RenderTarget *brt = atmGetBackRenderTarget();

        dc->setRenderTarget(m_rt_gauss0);
        m_sh_luminance->assign(dc);
        dc->setTexture(GLSL_COLOR_BUFFER, brt->getColorBuffer(GBUFFER_COLOR));
        dc->setTexture(GLSL_GLOW_BUFFER, m_rt_gbuffer->getColorBuffer(GBUFFER_GLOW));
        dc->setVertexArray(m_va_luminance);
        dc->draw(I3D_QUADS, 0, 16);
        dc->setRenderTarget(nullptr);
        stl::swap(m_rt_gauss0, m_rt_gauss1);
    }

    // 横ブラー
    for(uint32 i=0; i<2; ++i) {
        dc->setRenderTarget(m_rt_gauss0);
        m_sh_hblur->assign(dc);
        dc->setTexture(GLSL_COLOR_BUFFER, m_rt_gauss1->getColorBuffer(GBUFFER_COLOR));
        dc->setVertexArray(m_va_blur);
        dc->draw(I3D_QUADS, 0, 16);
        dc->setRenderTarget(nullptr);
        stl::swap(m_rt_gauss0, m_rt_gauss1);
    }

    // 縦ブラー
    {
        dc->setRenderTarget(m_rt_gauss0);
        m_sh_vblur->assign(dc);
        dc->setTexture(GLSL_COLOR_BUFFER, m_rt_gauss1->getColorBuffer(GBUFFER_COLOR));
        dc->setVertexArray(m_va_blur);
        dc->draw(I3D_QUADS, 0, 16);
        dc->setRenderTarget(nullptr);
        stl::swap(m_rt_gauss0, m_rt_gauss1);
    }

    // 加算
    dc->setViewport(*atmGetDefaultViewport());
    {
        RenderTarget *brt = atmGetBackRenderTarget();

        dc->setRenderTarget(brt);
        m_sh_composite->assign(dc);
        dc->setTexture(GLSL_COLOR_BUFFER, m_rt_gauss1->getColorBuffer(GBUFFER_COLOR));
        dc->setVertexArray(m_va_composite);
        dc->setBlendState(atmGetBlendState(BS_BLEND_ADD));
        dc->draw(I3D_QUADS, 0, 4);
        dc->setBlendState(atmGetBlendState(BS_NO_BLEND));
        dc->setTexture(GLSL_COLOR_BUFFER, nullptr);
        dc->setRenderTarget(nullptr);
    }
}


PassPostprocess_Fade::PassPostprocess_Fade()
{
    m_sh_fade       = atmGetShader(SH_FADE);
    m_va_quad       = atmGetVertexArray(VA_SCREEN_QUAD);
    m_ubo_fade      = atmGetUniformBuffer(UBO_FADE_PARAMS);

    m_loc_fade_param = m_sh_fade->getUniformBlockIndex("fade_params");
}

void PassPostprocess_Fade::beforeDraw()
{
    if(!atmGetGame()) { return; }

    float32 frame = (float32)atmGetFrame();
    if(frame > m_end_frame) { return; }

    vec4 diff = m_end_color-m_begin_color;
    float32 f = m_end_frame-m_begin_frame;
    float32 l = float32(frame-m_begin_frame)/f;
    m_params.Color = m_begin_color + diff*l;
}

void PassPostprocess_Fade::draw()
{
    if(m_params.Color.a==0.0f) { return; }

    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    MapAndWrite(dc, m_ubo_fade, &m_params, sizeof(m_params));

    RenderTarget *brt = atmGetBackRenderTarget();

    dc->setRenderTarget(brt);
    m_sh_fade->assign(dc);
    m_sh_fade->setUniformBlock(m_loc_fade_param, GLSL_FADE_BINDING, m_ubo_fade);
    dc->setVertexArray(m_va_quad);
    dc->setBlendState(atmGetBlendState(BS_BLEND_ALPHA));
    dc->draw(I3D_QUADS, 0, 4);
    dc->setBlendState(atmGetBlendState(BS_NO_BLEND));
    dc->setShader(nullptr);
    dc->setRenderTarget(nullptr);
}

void PassPostprocess_Fade::setFade(const vec4 &v, float32 frame)
{
    m_begin_color = m_params.Color;
    m_end_color = v;
    m_begin_frame = (float32)atmGetFrame();
    m_end_frame = m_begin_frame+frame;
}

} // namespace atm
