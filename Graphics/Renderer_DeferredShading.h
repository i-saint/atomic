#ifndef __atomic_Graphics_Renderer_DeferredShading__
#define __atomic_Graphics_Renderer_DeferredShading__
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


class PassDeferredShading_DirectionalLights : public IRenderer
{
private:
    typedef DirectionalLight light_t;
    typedef stl::vector<DirectionalLight> InstanceCont;
    InstanceCont    m_instances;

public:
    PassDeferredShading_DirectionalLights();
    void beforeDraw();
    void draw();

    void addInstance(const DirectionalLight& v);
};

class PassDeferredShading_PointLights : public IRenderer
{
public:

private:
    typedef stl::vector<PointLight> InstanceCont;
    InstanceCont    m_instances;

public:
    PassDeferredShading_PointLights();
    void beforeDraw();
    void draw();

    void addInstance(const PointLight& v) { m_instances.push_back(v); }
};

} // namespace atomic
#endif // __atomic_Graphics_Renderer_DeferredShading__
