#ifndef atomic_Graphics_Renderer_DeferredShading_h
#define atomic_Graphics_Renderer_DeferredShading_h
namespace atomic {


class PassDeferredShading_Bloodstain : public IRenderer
{
public:
    struct BloodstainParticleSet {
        mat4 transform;
        const BloodstainParticle *bsp_in;
        uint32 num_bsp;
    };

private:
    Buffer          *m_ibo_sphere;
    VertexArray     *m_va_sphere;
    Buffer          *m_vbo_bloodstain;
    AtomicShader    *m_sh;

    stl::vector<Task*>                  m_tasks;
    stl::vector<BloodstainParticleSet>  m_instances;
    stl::vector<BloodstainParticle>     m_particles;

    void resizeTasks(uint32 n);

public:
    PassDeferredShading_Bloodstain();
    ~PassDeferredShading_Bloodstain();
    void beforeDraw();
    void draw();

    void addBloodstainParticles(const mat4 &t, const BloodstainParticle *bsp, uint32 num_bsp);
};


class PassDeferredShading_Lights : public IRenderer
{
private:
    typedef stl::vector<DirectionalLight> DirectionalLights;
    typedef stl::vector<PointLight> PointLights;
    MultiresolutionParams   m_mr_params;
    DirectionalLights       m_directional_lights;
    PointLights             m_point_lights;
    uint32                  m_rendered_lights;

    void drawMultiResolution();
    void upsampling(int32 level);
    void debugShowResolution(int32 level);

    void updateConstantBuffers();
    void drawLights();
    void drawDirectionalLights();
    void drawPointLights();

public:
    PassDeferredShading_Lights();
    void beforeDraw();
    void draw();

    void addLight(const DirectionalLight& v);
    void addLight(const PointLight& v);

    MultiresolutionParams& getMultiresolutionParams() { return m_mr_params; }
};

} // namespace atomic
#endif // atomic_Graphics_Renderer_DeferredShading_h
