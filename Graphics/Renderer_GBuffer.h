#ifndef __atomic_Graphics_Renderer_GBuffer__
#define __atomic_Graphics_Renderer_GBuffer__
namespace atomic {


class PassGBuffer_SPH : public IRenderer
{
private:
    VertexArray     *m_va_cube;
    Buffer          *m_vbo_fluid;
    Buffer          *m_vbo_rigid;
    AtomicShader    *m_sh_fluid;
    AtomicShader    *m_sh_rigid;

    stl::vector<Task*>          m_tasks;
    stl::vector<PSetUpdateInfo> m_rupdateinfo;
    stl::vector<PSetParticle>   m_rparticles;
    stl::vector<PSetInstance>   m_rinstances;

    void resizeTasks(uint32 n);

public:
    PassGBuffer_SPH();
    ~PassGBuffer_SPH();
    void beforeDraw();
    void draw();

    void addPSetInstance(PSET_RID psid, const mat4 &t, const PSetInstance inst);
};

} // namespace atomic
#endif // __atomic_Graphics_Renderer_GBuffer__
