#ifndef __atomic_Graphics_Renderer_h__
#define __atomic_Graphics_Renderer_h__

#include "ResourceManager.h"

namespace atomic {


class IRenderer : public boost::noncopyable
{
public:
    virtual ~IRenderer() {}
    virtual void beforeDraw()=0;    // メインスレッドから、描画処理の前に呼ばれる。頂点データの用意などを行う
    virtual void draw()=0;          // 描画スレッドから呼ばれる。頂点データの GPU への転送、描画コマンド発行などを行う
};


class PassGBuffer_Cube;
class PassDeferred_PointLight;
class PassPostprocess_Bloom;


class AtomicRenderer : public boost::noncopyable
{
private:
    AtomicShader            *m_sh_out;

    RenderTargetGBuffer     *m_rt_gbuffer;
    RenderTargetDeferred    *m_rt_deferred;

    PassGBuffer_Cube        *m_renderer_cube;
    PassDeferred_PointLight *m_renderer_sphere_light;
    PassPostprocess_Bloom   *m_renderer_bloom;
    stl::vector<IRenderer*> m_renderers[PASS_END];

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

    PassGBuffer_Cube* getCubeRenderer()                 { return m_renderer_cube; }
    PassDeferred_PointLight* getSphereLightRenderer()   { return m_renderer_sphere_light; }
    const Viewport* getDefaultViewport() const          { return &m_default_viewport; }
    RenderStates* getRenderStates()                     { return &m_render_states; }
};

#define atomicGetCubeRenderer()         AtomicRenderer::getInstance()->getCubeRenderer()
#define atomicGetSphereLightRenderer()  AtomicRenderer::getInstance()->getSphereLightRenderer()
#define atomicGetDefaultViewport()      AtomicRenderer::getInstance()->getDefaultViewport()






class PassGBuffer_Cube : public IRenderer
{
private:
    struct InstanceInfo
    {
        stl::vector<float4> pos;
        stl::vector<float4> glow;
        stl::vector<float32> scale;

        void clear()
        {
            pos.clear();
            glow.clear();
            scale.clear();
        }

        void reserve(uint32 n)
        {
            pos.reserve(n);
            glow.reserve(n);
            scale.reserve(n);
        }
    };
    AtomicShader        *m_sh_gbuffer;
    ModelData           *m_model;
    VertexBufferObject  *m_vbo_fraction_pos;
    InstanceInfo        m_vfx;

public:
    PassGBuffer_Cube();
    void beforeDraw();  // メインスレッドから、描画処理の前に呼ばれる
    void draw();    // 描画スレッドから呼ばれる

    void pushVFXInstance(float4 v) { m_vfx.pos.push_back(v); }
};



class PassDeferred_PointLight : public IRenderer
{
private:
    stl::vector<float4> m_instance_pos;
    AtomicShader *m_shader;
    ModelData *m_model;
    VertexBufferObject *m_vbo_instance_pos;

public:
    PassDeferred_PointLight();
    void beforeDraw();
    void draw();

    void pushInstance(float4 v) { m_instance_pos.push_back(v); }
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
    AtomicShader            *m_sh_luminance;
    AtomicShader            *m_sh_hblur;
    AtomicShader            *m_sh_vblur;
    AtomicShader            *m_sh_composite;
    UniformBufferObject     *m_ubo_states;
    BloomStates             m_states;
    int                     m_loc_state;

public:
    PassPostprocess_Bloom();
    void beforeDraw();
    void draw();
};

} // namespace atomic
#endif // __atomic_Graphics_Renderer_h__
