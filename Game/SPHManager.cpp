#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/EntityQuery.h"
#include "Game/Entity.h"
#include "Game/SPHManager.h"
#include "Game/Collision.h"

namespace atomic {



class ComputeFluidParticle : public AtomicTask
{
private:
    PSET_RID m_psid;
    mat4 m_mat;
    thrust::host_vector<sphFluidParticle> m_fluid;

public:
    void setup(PSET_RID psid, const mat4 &m)
    {
        m_psid = psid;
        m_mat = m;
    }

    void exec()
    {
        const ParticleSet *rc = atomicGetParticleSet(m_psid);
        uint32 num_particles            = rc->getNumParticles();
        const PSetParticle *particles   = rc->getParticleData();
        m_fluid.resize(num_particles);

        simdmat4 t(m_mat);
        for(uint32 i=0; i<num_particles; ++i) {
            simdvec4 p(vec4(particles[i].position, 1.0f));
            simdvec4 n((vec4&)particles[i].normal);
            m_fluid[i].position = (float4&)glm::vec4_cast(t * p);
            m_fluid[i].velocity = make_float4(0.0f);
            m_fluid[i].energy   = 100.0f;
            m_fluid[i].density  = 0.0f;
            m_fluid[i].position.z = stl::max<float32>(m_fluid[i].position.z, 0.0f);
        }
    }

    const thrust::host_vector<sphFluidParticle>& getData() { return m_fluid; }
};

class SPHAsyncUpdateTask : public AtomicTask
{
private:
    SPHManager *m_obj;
    float32 m_arg;

public:
    SPHAsyncUpdateTask(SPHManager *v) : m_obj(v) {}
    void setup(float32 v) { m_arg=v; }

    void exec()
    {
        m_obj->taskAsyncupdate(m_arg);
    }
};


SPHManager::SPHManager()
    : m_current_fluid_task(0)
{
    m_asyncupdate_task = istNew(SPHAsyncUpdateTask)(this);
}

SPHManager::~SPHManager()
{
    for(uint32 i=0; i<m_fluid_tasks.size(); ++i) {
        istDelete(m_fluid_tasks[i]);
    }
    m_fluid_tasks.clear();
    istSafeDelete(m_asyncupdate_task);
}


void SPHManager::frameBegin()
{
    m_planes.clear();
    m_spheres.clear();
    m_boxes.clear();

    m_pgravity.clear();

    m_fluid.clear();
    m_current_fluid_task = 0;
}

void SPHManager::update(float32 dt)
{
    const thrust::host_vector<sphFluidMessage> &message = SPHGetFluidMessage();
    uint32 n = message.size();
    for(uint32 i=0; i<n; ++i) {
        const sphFluidMessage &m = message[i];
        if(IEntity *e = atomicGetEntity(m.to)) {
            atomicCall(e, eventFluid, &m);
        }
    }
}

void SPHManager::asyncupdate(float32 dt)
{
    static_cast<SPHAsyncUpdateTask*>(m_asyncupdate_task)->setup(dt);
    TaskScheduler::getInstance()->enqueue(m_asyncupdate_task);
}

void SPHManager::addRigid(const sphRigidPlane &s)   { m_planes.push_back(s); }
void SPHManager::addRigid(const sphRigidSphere &s)  { m_spheres.push_back(s); }
void SPHManager::addRigid(const sphRigidBox &s)     { m_boxes.push_back(s); }
void SPHManager::addForce(const sphForcePointGravity &v) { m_pgravity.push_back(v); }

void SPHManager::draw()
{
}

void SPHManager::frameEnd()
{
    m_asyncupdate_task->wait();
}


void SPHManager::copyParticlesToGL()
{
    m_asyncupdate_task->wait();
    SPHCopyToGL();
}

void SPHManager::taskAsyncupdate( float32 dt )
{
    atomicGetCollisionSet()->copyRigitsToGPU();

    for(uint32 i=0; i<m_current_fluid_task; ++i) {
        m_fluid_tasks[i]->wait();
        const thrust::host_vector<sphFluidParticle> &fluid = static_cast<ComputeFluidParticle*>(m_fluid_tasks[i])->getData();
        m_fluid.insert(m_fluid.end(), fluid.begin(), fluid.end());
    }
    DistanceField *df = atomicGetCollisionSet()->getDistanceField();

    SPHUpdateForce(m_pgravity);
    SPHUpdateRigids(m_planes, m_spheres, m_boxes);
#ifdef __atomic_enable_distance_field__
    SPHUpdateDistanceField((float4*)df->getDistances(), df->getEntities());
#endif // __atomic_enable_distance_field__
    SPHAddFluid(m_fluid);
    SPHUpdateFluid();
}


void SPHManager::addFluid( const sphFluidParticle *particles, uint32 num )
{
    m_fluid.insert(m_fluid.end(), particles, particles+num);
}

void SPHManager::addFluid(PSET_RID psid, const mat4 &t)
{
    while(m_fluid_tasks.size()<=m_current_fluid_task) {
        m_fluid_tasks.push_back( istNew(ComputeFluidParticle)() );
    }
    ComputeFluidParticle *task = static_cast<ComputeFluidParticle*>(m_fluid_tasks[m_current_fluid_task++]);
    task->setup(psid, t);
    TaskScheduler::getInstance()->enqueue(task);
}

} // namespace atomic
