#ifndef __atomic_Graphics_Renderer_DeferredShading__
#define __atomic_Graphics_Renderer_DeferredShading__
namespace atomic {


class PassDeferredShading_DirectionalLights : public IRenderer
{
private:
    typedef DirectionalLight light_t;
    typedef stl::vector<DirectionalLight> InstanceCont;
    InstanceCont    m_instances;
    VertexArray     *m_va_quad;
    VertexBuffer    *m_vbo_instance;
    AtomicShader    *m_shader;

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
    AtomicShader    *m_shader;
    IndexBuffer     *m_ibo_sphere;
    VertexArray     *m_va_sphere;
    VertexBuffer    *m_vbo_instance;

public:
    PassDeferredShading_PointLights();
    void beforeDraw();
    void draw();

    void addInstance(const PointLight& v) { m_instances.push_back(v); }
};

} // namespace atomic
#endif // __atomic_Graphics_Renderer_DeferredShading__
