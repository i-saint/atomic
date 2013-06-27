#ifndef atm_Graphics_Renderer_GBuffer_h
#define atm_Graphics_Renderer_GBuffer_h
namespace atm {



class PassGBuffer_Particle : public IRenderer
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

class PassGBuffer_Fluid : public IRenderer
{
public:
    PassGBuffer_Fluid();
    ~PassGBuffer_Fluid();
    void beforeDraw();
    void draw();

    void addPSetInstance(PSET_RID psid, const PSetInstance &inst);

private:
    ist::vector<PSetUpdateInfo> m_rupdateinfo;
    ist::vector<PSetParticle>   m_rparticles;
    ist::vector<PSetInstance>   m_rinstances;
};

} // namespace atm
#endif // atm_Graphics_Renderer_GBuffer_h
