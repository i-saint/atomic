#ifndef __atomic_Fraction_h__
#define __atomic_Fraction_h__

#include "GPGPU/SPH.cuh"
#include "Task.h"

namespace atomic {

class FractionGrid;
class Task_FractionUpdateAsync;
class Task_FractionDraw;


class Task_FractionUpdateAsync;


class FractionSet : boost::noncopyable
{
private:
    typedef Task_UpdateAsync<FractionSet> AsyncUpdateTask;

    AsyncUpdateTask                     *m_task_asyncupdate;
    SPHParticle                         m_particles[SPH_MAX_PARTICLE_NUM];
    SPHSphericalGravity                 m_sgravity[ SPH_MAX_SPHERICAL_GRAVITY_NUM ];
    thrust::host_vector<SPHParticle>    m_spawn;
    uint32          m_idgen;

public:
    FractionSet();
    ~FractionSet();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void update();
    void draw() const;
    void sync() const;
    void updateAsync();

    const SPHParticle* getFraction(uint32 i) const { return &m_particles[i]; }

public:
    void updateSPH();
    void processMessage();
};




} // namespace atomic
#endif // __atomic_Fraction_h__
