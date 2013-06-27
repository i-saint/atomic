#ifndef atm_Game_SPHManager_h
#define atm_Game_SPHManager_h

#include "Task.h"
#include "Graphics/ResourceManager.h"
#include "psym/psym.h"

namespace atm {

struct CollisionEntity;
typedef ist::raw_vector<psym::Particle> ParticleCont;

class FluidModule : public IAtomicGameModule
{
typedef IAtomicGameModule super;
public:
    struct AddFluidContext
    {
        PSET_RID psid;
        mat4 mat;
        uint32 index;
    };
    typedef ist::vector<AddFluidContext> AddFluidCtxCont;

public:
    FluidModule();
    ~FluidModule();

    void initialize();
    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    size_t copyParticlesToGL();
    void taskAsyncupdate(float32 dt);

    size_t getNumParticles() const;

    // rigid/force は毎フレームクリアされるので、毎フレーム突っ込む必要がある
    void addRigid(const CollisionEntity &v);
    void addForce(const psym::PointForce &v);
    void addFluid(psym::Particle *particles, uint32 num);
    void addFluid(PSET_RID psid, const mat4 &t);

private:
    psym::World         m_world;
    SFMT                m_rand;
    float32             m_gravity_strength;

    // 以下シリアライズ不要
    ist::Mutex          m_mutex_particles;
    ParticleCont        m_particles_to_gpu; // GPU 転送用
    uint32              m_current_fluid_task;
    ParticleCont        m_new_fluid;
    AddFluidCtxCont     m_new_fluid_ctx;


    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_world)
        istSerialize(m_rand)
        istSerialize(m_gravity_strength)
    )
};



} // namespace atm
#endif // atm_Game_SPHManager_h
