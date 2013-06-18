#ifndef atm_Graphics_Renderer_h
#define atm_Graphics_Renderer_h

#include "AtomicRenderingSystem.h"
#include "ResourceManager.h"
#include "Light.h"

namespace atm {


class IRenderer
{
istNonCopyable(IRenderer);
public:
    IRenderer() {}
    virtual ~IRenderer() {}
    virtual void beforeDraw() {}    // メインスレッドから、描画処理の前に呼ばれる。頂点データの用意などを行う
    virtual void draw()=0;          // 描画スレッドから呼ばれる。頂点データの GPU への転送、描画コマンド発行などを行う
};


class PassGBuffer_Fluid;
class PassGBuffer_Particle;
class PassDeferred_Bloodstain;
class PassDeferred_Lights;
class PassForward_Generic;
class PassForward_BackGround;
class PassPostprocess_Microscopic;
class PassPostprocess_FXAA;
class PassPostprocess_Bloom;
class PassPostprocess_Fade;
class SystemTextRenderer;
class PassHUD_DebugShowBuffer;

class PassForward_DistanceField; // for debug


class dpPatch AtomicRenderer
{
istNonCopyable(AtomicRenderer);
private:
    // shared resources
    VertexArray     *m_va_screenquad;
    AtomicShader    *m_sh_out;
    RenderTarget    *m_rt_gbuffer;
    RenderTarget    *m_rt_out[2];
    RenderTarget    *m_rt_prev_frame;

    // internal resources
    PassGBuffer_Fluid                   *m_pass_fluid;
    PassGBuffer_Particle                *m_pass_particle;
    PassDeferred_Bloodstain             *m_pass_bloodstain;
    PassDeferred_Lights                 *m_pass_lights;
    PassForward_Generic                 *m_pass_forward;
    PassForward_BackGround              *m_pass_bg;
    PassPostprocess_FXAA                *m_pass_fxaa;
    PassPostprocess_Bloom               *m_pass_bloom;
    PassPostprocess_Fade                *m_pass_fade;
    PassPostprocess_Microscopic         *m_pass_microscopic;
    PassForward_DistanceField    *m_pass_distance_field;
#ifdef atm_enable_gbuffer_viewer
    PassHUD_DebugShowBuffer             *m_debug_show_gbuffer;
#endif // atm_enable_gbuffer_viewer
    ist::vector<IRenderer*>             m_renderers[PASS_END];

    SystemTextRenderer                  *m_stext;

    Viewport            m_default_viewport;
    RenderStates        m_rstates3d;
    RenderStates        m_rstatesBG;
    RenderStates        m_rstates2d;

    uint32 m_frame;

private:
    static AtomicRenderer *s_inst;

    void passShadow();
    void passGBuffer();
    void passDeferredShading();
    void passForwardShading();
    void passPostprocess();
    void passHUD();
    void passOutput();

public:
    AtomicRenderer();
    ~AtomicRenderer();
    static AtomicRenderer* getInstance() { return s_inst; }
    static void initializeInstance();
    static void finalizeInstance();

    void beforeDraw();  // メインスレッドから、描画処理の前に呼ばれる
    void draw();        // 以下描画スレッドから呼ばれる

    const Viewport* getDefaultViewport() const  { return &m_default_viewport; }
    RenderStates* getRenderStates3D()           { return &m_rstates3d; }
    RenderStates* getRenderStatesBG()           { return &m_rstatesBG; }
    RenderStates* getRenderStates2D()           { return &m_rstates2d; }
    PassGBuffer_Fluid* getSPHPass()             { return m_pass_fluid; }
    PassGBuffer_Particle* getParticlePass()     { return m_pass_particle; }
    PassDeferred_Bloodstain* getBloodStainPass(){ return m_pass_bloodstain; }
    PassDeferred_Lights* getLightPass()         { return m_pass_lights; }
    PassForward_Generic* getForwardPass()       { return m_pass_forward; }
    PassPostprocess_Fade* getFader()            { return m_pass_fade; }
    SystemTextRenderer* getTextRenderer()       { return m_stext; }
    RenderTarget*   getFrontRenderTarget()      { return m_rt_out[0]; }
    RenderTarget*   getBackRenderTarget()       { return m_rt_out[1]; }
    RenderTarget*   getPrevBackbuffer()         { return m_rt_prev_frame; }
    void swapOutputRenderTarget()               { stl::swap(m_rt_out[0], m_rt_out[1]); }
    uint32 getRenderFrame() const               { return m_frame; }
};

#define atmGetRenderer()            AtomicRenderer::getInstance()
#define atmGetRenderStates()        atmGetRenderer()->getRenderStates3D()
#define atmGetRenderStatesBG()      atmGetRenderer()->getRenderStatesBG()
#define atmGetDefaultViewport()     atmGetRenderer()->getDefaultViewport()
#define atmGetBloodStainPass()      atmGetRenderer()->getBloodStainPass()
#define atmGetSPHPass()             atmGetRenderer()->getSPHPass()
#define atmGetParticlePass()        atmGetRenderer()->getParticlePass()
#define atmGetLightPass()           atmGetRenderer()->getLightPass()
#define atmGetForwardPass()         atmGetRenderer()->getForwardPass()
#define atmGetFader()               atmGetRenderer()->getFader()
#define atmGetTextRenderer()        atmGetRenderer()->getTextRenderer()

#define atmGetFrontRenderTarget()   atmGetRenderer()->getFrontRenderTarget()
#define atmGetBackRenderTarget()    atmGetRenderer()->getBackRenderTarget()
#define atmSwapOutputRenderTarget() atmGetRenderer()->swapOutputRenderTarget()
#define atmGetPrevBackbuffer()      atmGetRenderer()->getPrevBackbuffer()

#define atmGetRenderFrame()         atmGetRenderer()->getRenderFrame()


class SystemTextRenderer : public IRenderer
{
private:
    struct Text {
        wchar_t text[128];
        vec2 pos;
    };
    ist::vector<Text> m_texts;

public:
    SystemTextRenderer();
    void beforeDraw();
    void draw();

    void addText(const vec2 &pos, const char *text);
    void addText(const vec2 &pos, const wchar_t *text);
};

#ifdef atm_enable_gbuffer_viewer
class PassHUD_DebugShowBuffer : public IRenderer
{
private:
    void drawColorBuffer(const DebugShowBufferParams &params);
    void drawNormalBuffer(const DebugShowBufferParams &params);
    void drawPositionBuffer(const DebugShowBufferParams &params);
    void drawGlowBuffer(const DebugShowBufferParams &params);

public:
    RenderTarget *m_rt;
    RenderTarget *m_gbuffer;
    AtomicShader *m_sh_rgb;
    AtomicShader *m_sh_aaa;
    Buffer *m_ub_params;
    int32 m_loc_params;

    PassHUD_DebugShowBuffer();
    void draw();
};
#endif // atm_enable_gbuffer_viewer

} // namespace atm

#include "Renderer_GBuffer.h"
#include "Renderer_DeferredShading.h"
#include "Renderer_ForwardShading.h"
#include "Renderer_Postprocess.h"

#endif // atm_Graphics_Renderer_h
