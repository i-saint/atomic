#ifndef __atomic_Graphics_Renderer_h__
#define __atomic_Graphics_Renderer_h__

#include "GraphicResourceManager.h"

namespace atomic {


class Renderer : public boost::noncopyable
{
public:
    virtual ~Renderer() {}
    virtual void beforeDraw()=0;  // メインスレッドから、描画処理の前に呼ばれる
    virtual void draw()=0;    // 描画スレッドから呼ばれる
};


class PassGBuffer_Cube;
class PassDeferred_SphereLight;
class PassPostprocess_Bloom;


class AtomicRenderer : public boost::noncopyable
{
private:
    ShaderGBuffer *m_sh_gbuffer;
    ShaderDeferred *m_sh_deferred;
    ShaderOutput *m_sh_out;

    RenderTargetGBuffer *m_rt_gbuffer;
    RenderTargetDeferred *m_rt_deferred;

    PassGBuffer_Cube *m_renderer_cube;
    PassDeferred_SphereLight *m_renderer_sphere_light;
    PassPostprocess_Bloom *m_renderer_bloom;
    stl::vector<Renderer*> m_renderers[PASS_END];

    Viewport m_default_viewport;

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

    PassGBuffer_Cube* getCubeRenderer() { return m_renderer_cube; }
    PassDeferred_SphereLight* getSphereLightRenderer() { return m_renderer_sphere_light; }
    const Viewport* getDefaultViewport() const { return &m_default_viewport; }
};

#define atomicGetCubeRenderer()         AtomicRenderer::getInstance()->getCubeRenderer()
#define atomicGetSphereLightRenderer()  AtomicRenderer::getInstance()->getSphereLightRenderer()
#define atomicGetDefaultViewport()      AtomicRenderer::getInstance()->getDefaultViewport()


class PassGBuffer_Cube : public Renderer
{
private:
    struct InstanceInfo
    {
        stl::vector<XMVECTOR> pos;
        stl::vector<XMVECTOR> glow;
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
    ShaderGBuffer *m_sh_gbuffer;
    ModelData *m_model;
    VertexBufferObject *m_vbo_instance_pos;
    InstanceInfo m_fraction;
    InstanceInfo m_vfx;

public:
    PassGBuffer_Cube();
    void beforeDraw();  // メインスレッドから、描画処理の前に呼ばれる
    void draw();    // 描画スレッドから呼ばれる

    void pushFractionInstance(XMVECTOR v) { m_fraction.pos.push_back(v); }
    void pushVFXInstance(XMVECTOR v) { m_vfx.pos.push_back(v); }
};


class PassGBuffer_Octahedron : public Renderer
{
private:
    struct InstanceInfo
    {
        stl::vector<XMVECTOR> pos;
        stl::vector<XMVECTOR> glow;
        stl::vector<float32> scale;
        stl::vector<float32> time;
        stl::vector<float32> seed;

        void clear()
        {
            pos.clear();
            glow.clear();
            scale.clear();
            time.clear();
            seed.clear();
        }

        void reserve(uint32 n)
        {
            pos.reserve(n);
            glow.reserve(n);
            scale.reserve(n);
            time.clear();
            seed.clear();
        }
    };
    ShaderGBuffer_Octahedron *m_sh_gbuffer;
    ModelData *m_model;
    VertexBufferObject *m_vbo_instance_pos;
    InstanceInfo m_bullets;

public:
    PassGBuffer_Octahedron();
    void beforeDraw();
    void draw();

    void pushInstance(XMVECTOR v) { m_bullets.pos.push_back(v); }
};



class PassDeferred_SphereLight : public Renderer
{
private:
    stl::vector<XMVECTOR> m_instance_pos;
    ShaderDeferred *m_sh_deferred;
    ModelData *m_model;
    VertexBufferObject *m_vbo_instance_pos;

public:
    PassDeferred_SphereLight();
    void beforeDraw();
    void draw();

    void pushInstance(XMVECTOR v) { m_instance_pos.push_back(v); }
};


class PassPostprocess_Bloom : public Renderer
{
private:
    RenderTargetDeferred *m_rt_deferred;
    RenderTargetGauss *m_rt_gauss0;
    RenderTargetGauss *m_rt_gauss1;
    ShaderBloom *m_sh_bloom;

public:
    PassPostprocess_Bloom();
    void beforeDraw();
    void draw();
};

} // namespace atomic
#endif // __atomic_Graphics_Renderer_h__
