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


class SPHManager : boost::noncopyable
{
private:
    thrust::host_vector<sphRigidInstance>   m_rigids;
    thrust::host_vector<sphRigidSphere>     m_spheres;
    thrust::host_vector<sphRigidBox>        m_boxes;
    thrust::host_vector<sphForcePointGravity>   m_pgravity;

    void addRigidInstance(CB_RID cid, EntityHandle h, const mat4 &m);

public:
    SPHManager();
    ~SPHManager();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void updateBegin(float32 dt);
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw() const;

    void addRigidSphere(CB_RID cid, EntityHandle h, const mat4 &m, const sphRigidSphere &s);
    void addRigidBox(CB_RID cid, EntityHandle h, const mat4 &m, const sphRigidBox &s);
    void addPointGravity();
};




} // namespace atomic
#endif // __atomic_Game_SPHManager__
