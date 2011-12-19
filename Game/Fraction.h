#ifndef __atomic_Fraction_h__
#define __atomic_Fraction_h__

#include "GPGPU/SPH.cuh"
#include "Task.h"
#include "Graphics/ResourceManager.h"

namespace atomic {

class FractionGrid;
class Task_FractionUpdateAsync;
class Task_FractionDraw;


class Task_FractionUpdateAsync;


class FractionSet : boost::noncopyable
{
private:
    typedef Task_UpdateAsync<FractionSet> AsyncUpdateTask;

    AsyncUpdateTask                         *m_task_asyncupdate;
    SPHFluidParticle                        m_particles[SPH_MAX_FLUID_PARTICLES];
    SPHSphericalGravity                     m_sgravity[ SPH_MAX_SPHERICAL_GRAVITY_NUM ];
    thrust::host_vector<SPHFluidParticle>   m_spawn;

    stl::vector<SPHCharacterInstance>   m_matrices[CB_END];

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

    const SPHFluidParticle* getFraction(uint32 i) const { return &m_particles[i]; }

    void addMatrix(CB_RID id, EntityHandle h, const mat4& m)
    {
        SPHCharacterInstance tmp;
        tmp.handle = h;
        tmp.transform = m;
        m_matrices[id].push_back( tmp );
    }

public:
    void updateSPH();
    void processMessage();
};




} // namespace atomic
#endif // __atomic_Fraction_h__
