#ifndef __atomic_Graphics_Renderer_h__
#define __atomic_Graphics_Renderer_h__

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


class PassGBuffer_Fraction;
class PassDeferred_DirectionalLights;
class PassDeferred_PointLights;
class PassPostprocess_Bloom;


class AtomicRenderer : public boost::noncopyable
{
private:
    // shared resources
    VertexArray             *m_va_screenquad;
    AtomicShader            *m_sh_out;
    RenderTargetGBuffer     *m_rt_gbuffer;
    RenderTargetDeferred    *m_rt_deferred;

    // internal resources
    PassGBuffer_Fraction            *m_renderer_cube;
    PassDeferred_DirectionalLights  *m_renderer_directional_light;
    PassDeferred_PointLights        *m_renderer_sphere_light;
    PassPostprocess_Bloom           *m_renderer_bloom;
    stl::vector<IRenderer*>         m_renderers[PASS_END];

    Viewport        m_default_viewport;
    RenderStates    m_render_states;

private:
    static AtomicRenderer *s_inst;

    void pass_Shadow();
    void pass_GBuffer();
    void pass_Deferred();
    void pass_Forward();
    void pass_Postprocess();
    void pass_UI();
    void pass_Output();

public:
    AtomicRenderer();
    ~AtomicRenderer();
    static AtomicRenderer* getInstance() { return s_inst; }
    static void initializeInstance();
    static void finalizeInstance();

    void beforeDraw();  // メインスレッドから、描画処理の前に呼ばれる
    void draw();        // 以下描画スレッドから呼ばれる

    PassGBuffer_Fraction* getCubeRenderer()             { return m_renderer_cube; }
    PassDeferred_DirectionalLights* getDirectionalLightRenderer() { return m_renderer_directional_light; }
    PassDeferred_PointLights* getSphereLightRenderer()  { return m_renderer_sphere_light; }
    const Viewport* getDefaultViewport() const          { return &m_default_viewport; }
    RenderStates* getRenderStates()                     { return &m_render_states; }
};

#define atomicGetCubeRenderer()             AtomicRenderer::getInstance()->getCubeRenderer()
#define atomicGetDirectionalLightRenderer() AtomicRenderer::getInstance()->getDirectionalLightRenderer()
#define atomicGetSphereLightRenderer()      AtomicRenderer::getInstance()->getSphereLightRenderer()
#define atomicGetDefaultViewport()          AtomicRenderer::getInstance()->getDefaultViewport()






class PassGBuffer_Fraction : public IRenderer
{
private:
    AtomicShader        *m_sh_gbuffer;
    VertexArray         *m_va_fraction;
    VertexBufferObject  *m_vbo_instance;

public:
    PassGBuffer_Fraction();
    void beforeDraw();  // メインスレッドから、描画処理の前に呼ばれる
    void draw();    // 描画スレッドから呼ばれる
};



class PassDeferred_DirectionalLights : public IRenderer
{
private:
    typedef DirectionalLight light_t;
    typedef stl::vector<DirectionalLight> InstanceCont;
    InstanceCont        m_instances;
    AtomicShader        *m_shader;
    VertexArray         *m_va_quad;
    VertexBufferObject  *m_vbo_instance;

public:
    PassDeferred_DirectionalLights();
    void beforeDraw();
    void draw();

    void pushInstance(const DirectionalLight& v);
};

class PassDeferred_PointLights : public IRenderer
{
public:
    typedef PointLight light_t;
    struct Light
    {
        vec4 position;
    };

private:
    typedef stl::vector<Light> InstanceCont;
    InstanceCont        m_instances;
    AtomicShader        *m_shader;
    IndexBufferObject   *m_ibo_sphere;
    VertexArray         *m_va_sphere;
    VertexBufferObject  *m_vbo_instance;

public:
    PassDeferred_PointLights();
    void beforeDraw();
    void draw();

    void pushInstance(const Light& v) { m_instances.push_back(v); }
};

//
//class PassPostprocess_FXAA : public Renderer
//{
//private:
//    ColorBuffer *m_rt_RGBL;
//    ShaderFXAA *m_sh_FXAA;
//
//public:
//    PassPostprocess_FXAA();
//    void beforeDraw();
//    void draw();
//};

class PassPostprocess_Bloom : public IRenderer
{
private:
    RenderTargetDeferred    *m_rt_deferred;
    RenderTargetGauss       *m_rt_gauss0;
    RenderTargetGauss       *m_rt_gauss1;
    VertexArray             *m_va_luminance;
    VertexArray             *m_va_blur;
    VertexArray             *m_va_composite;
    AtomicShader            *m_sh_luminance;
    AtomicShader            *m_sh_hblur;
    AtomicShader            *m_sh_vblur;
    AtomicShader            *m_sh_composite;
    UniformBufferObject     *m_ubo_states;

public:
    PassPostprocess_Bloom();
    void beforeDraw();
    void draw();
};

} // namespace atomic
#endif // __atomic_Graphics_Renderer_h__
