#ifndef atomic_Game_SPHManager_h
#define atomic_Game_SPHManager_h

#include "Task.h"
#include "Graphics/ResourceManager.h"
#include "psym/psym.h"

namespace atomic {

struct CollisionEntity;

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
#endif // atomic_Game_SPHManager_h
