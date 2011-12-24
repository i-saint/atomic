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
    sphForcePointGravity                     m_sgravity[ SPH_MAX_SPHERICAL_GRAVITY_NUM ];

    thrust::host_vector<sphRigidInstance>   m_rigids;

public:
    SPHManager();
    ~SPHManager();

    void initialize();
    void serialize(Serializer& s) const;
    void deserialize(Deserializer& s);

    void updateBegin(float32 dt);
    void update(float32 dt);
    void updateAsync(float32 dt);
    void draw() const;

    void addRigid(CB_RID cid, EntityHandle h, const mat4& m);

public:
    void updateSPH();
    void processMessage();
};




} // namespace atomic
#endif // __atomic_Game_SPHManager__
