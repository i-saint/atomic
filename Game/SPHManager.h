#ifndef __atomic_Game_SPHManager__
#define __atomic_Game_SPHManager__

#include "GPGPU/SPH.cuh"
#include "Task.h"
#include "Graphics/ResourceManager.h"

namespace atomic {

class FractionGrid;
class Task_FractionUpdateAsync;
class Task_FractionDraw;
class Task_FractionUpdateAsync;


class SPHManager : public AtomicGameModule
{
private:
    thrust::host_vector<sphRigidPlane>          m_planes;
    thrust::host_vector<sphRigidSphere>         m_spheres;
    thrust::host_vector<sphRigidBox>            m_boxes;
    thrust::host_vector<sphForcePointGravity>   m_pgravity;
    thrust::host_vector<sphFluidParticle>       m_fluid;
    stl::vector<Task*>  m_fluid_tasks;
    Task*               m_asyncupdate_task;
    uint32              m_current_fluid_task;

public:
    SPHManager();
    ~SPHManager();

    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    void copyParticlesToGL();
    void taskAsyncupdate(float32 dt);

    // rigid/force は毎フレームクリアされるので、毎フレーム突っ込む必要がある
    void addRigid(const sphRigidPlane &s);
    void addRigid(const sphRigidSphere &s);
    void addRigid(const sphRigidBox &s);
    void addForce(const sphForcePointGravity &v);
    void addFluid(const sphFluidParticle *particles, uint32 num);
    void addFluid(PSET_RID psid, const mat4 &t);
};




} // namespace atomic
#endif // __atomic_Game_SPHManager__
