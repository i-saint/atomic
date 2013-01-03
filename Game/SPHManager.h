#ifndef __atomic_Game_SPHManager__
#define __atomic_Game_SPHManager__

#include "Task.h"
#include "Graphics/ResourceManager.h"
#include "psym/psym.h"

namespace atomic {
//
//class FractionGrid;
//class Task_FractionUpdateAsync;
//class Task_FractionDraw;
//class Task_FractionUpdateAsync;
//
//class SPHManager : public IAtomicGameModule
//{
//private:
//    thrust::host_vector<sphRigidPlane>          m_planes;
//    thrust::host_vector<sphRigidSphere>         m_spheres;
//    thrust::host_vector<sphRigidBox>            m_boxes;
//    thrust::host_vector<sphForcePointGravity>   m_pgravity;
//    thrust::host_vector<sphFluidParticle>       m_fluid;
//    stl::vector<Task*>  m_fluid_tasks;
//    Task*               m_asyncupdate_task;
//    uint32              m_current_fluid_task;
//
//public:
//    SPHManager();
//    ~SPHManager();
//
//    void serialize(Serializer& s) const;
//    void deserialize(Deserializer& s);
//
//    void frameBegin();
//    void update(float32 dt);
//    void asyncupdate(float32 dt);
//    void draw();
//    void frameEnd();
//
//    void copyParticlesToGL();
//    void taskAsyncupdate(float32 dt);
//
//    // rigid/force は毎フレームクリアされるので、毎フレーム突っ込む必要がある
//    void addRigid(const sphRigidPlane &s);
//    void addRigid(const sphRigidSphere &s);
//    void addRigid(const sphRigidBox &s);
//    void addForce(const sphForcePointGravity &v);
//    void addFluid(const sphFluidParticle *particles, uint32 num);
//    void addFluid(PSET_RID psid, const mat4 &t);
//};

struct CollisionEntity;
class Task_FractionUpdateAsync2;
class Task_FractionDraw2;
class Task_FractionUpdateAsync2;

class SPHManager : public IAtomicGameModule
{
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

    size_t copyParticlesToGL();
    void taskAsyncupdate(float32 dt);

    size_t getNumParticles() const;

    // rigid/force は毎フレームクリアされるので、毎フレーム突っ込む必要がある
    void addRigid(const CollisionEntity &v);
    void addForce(const psym::PointForce &v);
    void addFluid(psym::Particle *particles, uint32 num);
    void addFluid(PSET_RID psid, const mat4 &t);

private:
    psym::World m_world;
    ist::Mutex m_mutex_particles;
    stl::vector<psym::Particle> m_particles; // GPU 転送用
    stl::vector<Task*>  m_fluid_tasks;
    Task*               m_asyncupdate_task;
    uint32              m_current_fluid_task;
    SFMT                m_rand;
};



} // namespace atomic
#endif // __atomic_Game_SPHManager__
