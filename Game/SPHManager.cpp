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
#include "Game/Collision.h"
#include "Game/SPHManager.h"
#include "Collision.h"

namespace atomic {


class ComputeFluidParticle : public AtomicTask
{
private:
    PSET_RID m_psid;
    mat4 m_mat;
    stl::vector<psym::Particle> m_fluid;

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

        simdvec4 zero(vec4(0.0f, 0.0f, 0.0f, 0.0f));
        simdmat4 t(m_mat);
        for(uint32 i=0; i<num_particles; ++i) {
            simdvec4 p(vec4(particles[i].position, 1.0f));
            istAlign(16) vec4 pos = glm::vec4_cast(t * p);
            pos.z = stl::max<float32>(pos.z, 0.0f);
            m_fluid[i].position = reinterpret_cast<const psym::simdvec4&>(pos);
            m_fluid[i].velocity = reinterpret_cast<const psym::simdvec4&>(zero);
        }
    }

    stl::vector<psym::Particle>& getData() { return m_fluid; }
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
    , m_asyncupdate_task(NULL)
{
    m_asyncupdate_task = istNew(SPHAsyncUpdateTask)(this);
    m_rand.initialize(0);
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
    m_world.clearRigidsAndForces();
    m_current_fluid_task = 0;
}

void SPHManager::update( float32 dt )
{
    const psym::Particle *feedback = m_world.getParticles();
    uint32 n = m_world.getNumParticles();
    for(uint32 i=0; i<n; ++i) {
        const psym::Particle &m = feedback[i];
        if(IEntity *e = atomicGetEntity(m.hit_to)) {
            atomicCall(e, eventFluid, &m);
        }
    }

    {
        // 床
        psym::RigidPlane plane;
        plane.id = 0;
        plane.bb.bl_x = -PSYM_GRID_SIZE;
        plane.bb.bl_y = -PSYM_GRID_SIZE;
        plane.bb.bl_z = -PSYM_GRID_SIZE;
        plane.bb.ur_x =  PSYM_GRID_SIZE;
        plane.bb.ur_y =  PSYM_GRID_SIZE;
        plane.bb.ur_z =  PSYM_GRID_SIZE;
        plane.nx = 0.0f;
        plane.ny = 0.0f;
        plane.nz = 1.0f;
        plane.distance = 0.0f;
        m_world.addRigid(plane);
    }
    {
        // 重力
        psym::DirectionalForce grav;
        grav.nx = 0.0f;
        grav.ny = 0.0f;
        grav.nz = -1.0f;
        grav.strength = 10.0f;
        m_world.addForce(grav);
    }
    atomicGetCollisionSet()->copyRigitsToPSym();
}

void SPHManager::asyncupdate( float32 dt )
{
    m_particles.clear();
    m_mutex_particles.lock();

    static_cast<SPHAsyncUpdateTask*>(m_asyncupdate_task)->setup(dt);
    TaskScheduler::getInstance()->enqueue(m_asyncupdate_task);
}

void SPHManager::draw()
{

}

void SPHManager::frameEnd()
{
    m_asyncupdate_task->wait();
}

size_t SPHManager::copyParticlesToGL()
{
    ist::ScopedLock<ist::Mutex> l(m_mutex_particles);
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    Buffer *vb = atomicGetVertexBuffer(VBO_FLUID_PARTICLES);
    MapAndWrite(dc, vb, &m_particles[0], m_particles.size()*sizeof(psym::Particle));
    return m_particles.size();
}

void SPHManager::taskAsyncupdate( float32 dt )
{
    for(uint32 i=0; i<m_current_fluid_task; ++i) {
        m_fluid_tasks[i]->wait();
        ComputeFluidParticle *cfp = static_cast<ComputeFluidParticle*>(m_fluid_tasks[i]);
        stl::vector<psym::Particle> &fluid = cfp->getData();
        addFluid(&fluid[0], fluid.size());
    }
    m_particles.insert(m_particles.end(), m_world.getParticles(), m_world.getParticles()+m_world.getNumParticles());
    m_mutex_particles.unlock();
    m_world.update(dt);
}

size_t SPHManager::getNumParticles() const
{
    return m_world.getNumParticles();
}

void SPHManager::addRigid(const CollisionEntity &v)
{
    switch(v.getShape()) {
    case CS_SPHERE:
        {
            const CollisionSphere &src = static_cast<const CollisionSphere&>(v);
            psym::RigidSphere dst;
            dst.id = src.getGObjHandle();
            dst.bb.bl_x = src.bb.bl.x;
            dst.bb.bl_y = src.bb.bl.y;
            dst.bb.bl_z = src.bb.bl.z;
            dst.bb.ur_x = src.bb.ur.x;
            dst.bb.ur_y = src.bb.ur.y;
            dst.bb.ur_z = src.bb.ur.z;
            dst.x = src.pos_r.x;
            dst.y = src.pos_r.y;
            dst.z = src.pos_r.z;
            dst.radius = src.pos_r.w;
            m_world.addRigid(dst);
        }
        break;

    case CS_PLANE:
        {
            const CollisionPlane &src = static_cast<const CollisionPlane&>(v);
            psym::RigidPlane dst;
            dst.id = v.getGObjHandle();
            dst.bb.bl_x = src.bb.bl.x;
            dst.bb.bl_y = src.bb.bl.y;
            dst.bb.bl_z = src.bb.bl.z;
            dst.bb.ur_x = src.bb.ur.x;
            dst.bb.ur_y = src.bb.ur.y;
            dst.bb.ur_z = src.bb.ur.z;
            dst.nx = src.plane.x;
            dst.ny = src.plane.y;
            dst.nz = src.plane.z;
            dst.distance = src.plane.w;
            m_world.addRigid(dst);
        }
        break;

    case CS_BOX:
        {
            const CollisionBox &src = static_cast<const CollisionBox&>(v);
            psym::RigidBox dst;
            dst.id = v.getGObjHandle();
            dst.bb.bl_x = src.bb.bl.x;
            dst.bb.bl_y = src.bb.bl.y;
            dst.bb.bl_z = src.bb.bl.z;
            dst.bb.ur_x = src.bb.ur.x;
            dst.bb.ur_y = src.bb.ur.y;
            dst.bb.ur_z = src.bb.ur.z;
            dst.x = src.position.x;
            dst.y = src.position.y;
            dst.z = src.position.z;
            for(size_t i=0; i<6; ++i) {
                dst.planes[i].nx = src.planes[i].x;
                dst.planes[i].ny = src.planes[i].y;
                dst.planes[i].nz = src.planes[i].z;
                dst.planes[i].distance = src.planes[i].w;
            }
            m_world.addRigid(dst);
        }
        break;

    default:
        istAssert(false, "unknown collision shape\n");
    }
}

void SPHManager::addForce( const psym::PointForce &v )
{
    m_world.addForce(v);
}

void SPHManager::addFluid( psym::Particle *particles, uint32 num )
{
    const float32 enery_base = 2700.0f;
    const float32 enery_diffuse = 300.0f;
    for(uint32 i=0; i<num; ++i) {
        particles[i].energy = enery_base + (m_rand.genFloat32()*enery_diffuse);
        particles[i].density = 0.0f;
    }
    m_world.addParticles(particles, num);
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
