#ifndef atomic_Graphics_Renderer_GBuffer_h
#define atomic_Graphics_Renderer_GBuffer_h
namespace atomic {

class UpdateRigidParticle
{
public:
    UpdateRigidParticle(const PSetUpdateInfo &ri, PSetParticle *p);
    void exec();

private:
    const PSetUpdateInfo *m_rinst;
    PSetParticle *m_particles;
};


class PassGBuffer_Particle : public IRenderer
{
public:
    PassGBuffer_Particle();
    ~PassGBuffer_Particle();
    void beforeDraw();
    void draw();

    void addParticle(const IndivisualParticle *particles, uint32 num);

private:
    VertexArray     *m_va_cube;
    Buffer          *m_vbo;
    AtomicShader    *m_sh;
    stl::vector<IndivisualParticle> m_particles;
};

class PassGBuffer_Fluid : public IRenderer
{
public:
    PassGBuffer_Fluid();
    ~PassGBuffer_Fluid();
    void beforeDraw();
    void draw();

    void addPSetInstance(PSET_RID psid, const PSetInstance inst);

private:
    VertexArray     *m_va_cube;
    Buffer          *m_vbo_fluid;
    Buffer          *m_vbo_rigid;
    AtomicShader    *m_sh_fluid;
    AtomicShader    *m_sh_rigid;

    stl::vector<PSetUpdateInfo> m_rupdateinfo;
    stl::vector<PSetParticle>   m_rparticles;
    stl::vector<PSetInstance>   m_rinstances;
    stl::vector<UpdateRigidParticle>    m_updater;
};


class PassGBuffer_BG : public IRenderer
{
public:
    PassGBuffer_BG();
    ~PassGBuffer_BG();
    void beforeDraw();
    void draw();

private:
    bool m_enabled;
};

} // namespace atomic
#endif // atomic_Graphics_Renderer_GBuffer_h
