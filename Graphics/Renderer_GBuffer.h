#ifndef __atomic_Graphics_Renderer_GBuffer__
#define __atomic_Graphics_Renderer_GBuffer__
namespace atomic {

class UpdateRigidParticle
{
private:
    const PSetUpdateInfo *m_rinst;
    PSetParticle *m_particles;

public:
    UpdateRigidParticle(const PSetUpdateInfo &ri, PSetParticle *p);
    void exec();
};


class PassGBuffer_Particle : public IRenderer
{
private:
    VertexArray     *m_va_cube;
    Buffer          *m_vbo;
    AtomicShader    *m_sh;
    stl::vector<IndivisualParticle> m_particles;

public:
    PassGBuffer_Particle();
    ~PassGBuffer_Particle();
    void beforeDraw();
    void draw();

    void addParticle(const IndivisualParticle *particles, uint32 num);
};

class PassGBuffer_SPH : public IRenderer
{
private:
    VertexArray     *m_va_cube;
    Buffer          *m_vbo_fluid;
    Buffer          *m_vbo_rigid;
    AtomicShader    *m_sh_fluid;
    AtomicShader    *m_sh_rigid;

    stl::vector<PSetUpdateInfo> m_rupdateinfo;
    stl::vector<PSetParticle>   m_rparticles;
    stl::vector<PSetInstance>   m_rinstances;
    stl::vector<Task*>                  m_tasks;
    stl::vector<UpdateRigidParticle>    m_updater;

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
