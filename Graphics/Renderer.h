#ifndef __atomic_Graphics_Renderer__
#define __atomic_Graphics_Renderer__

#include "ResourceManager.h"
#include "Light.h"

namespace atomic {


class IRenderer : public boost::noncopyable
{
public:
    virtual ~IRenderer() {}
    virtual void beforeDraw()=0;    // メインスレッドから、描画処理の前に呼ばれる。頂点データの用意などを行う
    virtual void draw()=0;          // 描画スレッドから呼ばれる。頂点データの GPU への転送、描画コマンド発行などを行う
};


class PassGBuffer_SPH;
class PassGBuffer_ParticleSet;
class PassDeferredShading_DirectionalLights;
class PassDeferredShading_PointLights;
class PassPostprocess_Microscopic;
class PassPostprocess_FXAA;
class PassPostprocess_Bloom;
class PassPostprocess_Fade;
class SystemTextRenderer;

class PassForwardShading_DistanceField; // for debug


class AtomicRenderer : public boost::noncopyable
{
private:
    // shared resources
    VertexArray     *m_va_screenquad;
    AtomicShader    *m_sh_out;
    RenderTarget    *m_rt_gbuffer;
    RenderTarget    *m_rt_out[2];

    // internal resources
    PassGBuffer_SPH                         *m_renderer_sph;
    PassDeferredShading_DirectionalLights   *m_renderer_dir_lights;
    PassDeferredShading_PointLights         *m_renderer_point_lights;
    PassPostprocess_FXAA                    *m_renderer_fxaa;
    PassPostprocess_Bloom                   *m_renderer_bloom;
    PassPostprocess_Fade                    *m_renderer_fade;
    PassPostprocess_Microscopic             *m_renderer_microscopic;
    PassForwardShading_DistanceField        *m_renderer_distance_field;
    stl::vector<IRenderer*>                 m_renderers[PASS_END];

    SystemTextRenderer                      *m_stext;

    Viewport   m_default_viewport;
    RenderStates    m_render_states;

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

    const Viewport* getDefaultViewport() const                      { return &m_default_viewport; }
    RenderStates* getRenderStates()                                 { return &m_render_states; }
    PassGBuffer_SPH* getSPHRenderer()                               { return m_renderer_sph; }
    PassDeferredShading_DirectionalLights* getDirectionalLights()   { return m_renderer_dir_lights; }
    PassDeferredShading_PointLights* getPointLights()               { return m_renderer_point_lights; }
    PassPostprocess_Fade* getFader()                                { return m_renderer_fade; }
    SystemTextRenderer* getSystemTextRenderer()                     { return m_stext; }

    RenderTarget*   getFrontRenderTarget() { return m_rt_out[0]; }
    RenderTarget*   getBackRenderTarget() { return m_rt_out[1]; }
    void swapOutputRenderTarget()   { std::swap(m_rt_out[0], m_rt_out[1]); }
};

#define atomicGetRenderer()             AtomicRenderer::getInstance()
#define atomicGetRenderStates()         atomicGetRenderer()->getRenderStates()
#define atomicGetDefaultViewport()      atomicGetRenderer()->getDefaultViewport()
#define atomicGetSPHRenderer()          atomicGetRenderer()->getSPHRenderer()
#define atomicGetDirectionalLights()    atomicGetRenderer()->getDirectionalLights()
#define atomicGetPointLights()          atomicGetRenderer()->getPointLights()
#define atomicGetFader()                atomicGetRenderer()->getFader()
#define atomicGetSystemTextRenderer()   atomicGetRenderer()->getSystemTextRenderer()

#define atomicGetFrontRenderTarget()    atomicGetRenderer()->getFrontRenderTarget()
#define atomicGetBackRenderTarget()     atomicGetRenderer()->getBackRenderTarget()
#define atomicSwapOutputRenderTarget()  atomicGetRenderer()->swapOutputRenderTarget()



class SystemTextRenderer : public IRenderer
{
private:
    struct Text {
        char text[128];
        ivec2 pos;
    };
    stl::vector<Text> m_texts;

public:
    SystemTextRenderer();
    void beforeDraw();
    void draw();

    void addText(const ivec2 &pos, const char *text);
};

} // namespace atomic

#include "Renderer_GBuffer.h"
#include "Renderer_DeferredShading.h"
#include "Renderer_ForwardShading.h"
#include "Renderer_Postprocess.h"

#endif // __atomic_Graphics_Renderer__
