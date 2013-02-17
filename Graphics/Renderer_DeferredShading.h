#ifndef atomic_Graphics_Renderer_DeferredShading_h
#define atomic_Graphics_Renderer_DeferredShading_h
namespace atomic {


class PassDeferredShading_Bloodstain : public IRenderer
{
public:
    // 血痕パーティクル群
    // 既存のオブジェクトにへばりついているパーティクルとして表現される。
    // transform はその親オブジェクト。
    struct BloodstainParticleSet
    {
        mat4 transform;
        uint32 num_bsp;
        const BloodstainParticle *bp_in;
        BloodstainParticle *bp_out;
    };

public:
    PassDeferredShading_Bloodstain();
    ~PassDeferredShading_Bloodstain();
    void beforeDraw();
    void draw();

    void addBloodstainParticles(const mat4 &t, const BloodstainParticle *bsp, uint32 num_bsp);

private:
    Buffer          *m_ibo_sphere;
    VertexArray     *m_va_sphere;
    Buffer          *m_vbo_bloodstain;
    AtomicShader    *m_sh;
    ist::raw_vector<BloodstainParticleSet>  m_instances;
    ist::raw_vector<BloodstainParticle>     m_particles;
};


class PassDeferredShading_Lights : public IRenderer
{
public:
    PassDeferredShading_Lights();
    void beforeDraw();
    void draw();

    void addLight(const DirectionalLight& v);
    void addLight(const PointLight& v);

    MultiresolutionParams& getMultiresolutionParams() { return m_mr_params; }

private:
    void drawMultiResolution();
    void upsampling(int32 level);
    void debugShowResolution(int32 level);

    void updateConstantBuffers();
    void drawLights();
    void drawDirectionalLights();
    void drawPointLights();

    typedef ist::vector<DirectionalLight> DirectionalLights;
    typedef ist::vector<PointLight> PointLights;
    MultiresolutionParams   m_mr_params;
    DirectionalLights       m_directional_lights;
    PointLights             m_point_lights;
    uint32                  m_rendered_lights;
};

} // namespace atomic
#endif // atomic_Graphics_Renderer_DeferredShading_h
