#ifndef atm_Graphics_Renderer_GBuffer_h
#define atm_Graphics_Renderer_GBuffer_h
namespace atm {



class atmAPI PassGBuffer_Particle : public IRenderer
{
public:
    PassGBuffer_Particle();
    ~PassGBuffer_Particle();
    void beforeDraw();
    void draw();

    void addParticle(const SingleParticle *particles, uint32 num);

private:
    ist::vector<SingleParticle> m_particles;
};

class atmAPI PassGBuffer_Fluid : public IRenderer
{
public:
    PassGBuffer_Fluid();
    ~PassGBuffer_Fluid();
    void beforeDraw();
    void draw();

    void addParticles(PSET_RID psid, const PSetInstance &inst, uint32 n=0);
    void addParticlesSolid(PSET_RID psid, const PSetInstance &inst, uint32 n=0);

    static bool culling(PSET_RID psid, const PSetInstance &inst);
    static void drawParticleSets(PSetDrawData &pdd);

private:
    void drawFluid();

    PSetDrawData m_rigid_sp;
    PSetDrawData m_rigid_so;
};

} // namespace atm
#endif // atm_Graphics_Renderer_GBuffer_h
