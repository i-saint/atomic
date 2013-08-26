#include "atmPCH.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "Engine/Graphics/Renderer.h"
#include "Engine/Game/Message.h"
#include "Engine/Game/AtomicGame.h"
#include "Engine/Game/World.h"
#include "Engine/Game/EntityModule.h"
#include "Engine/Game/EntityQuery.h"
#include "Engine/Game/CollisionModule.h"
#include "Engine/Game/FluidModule.h"
#include "CollisionModule.h"

namespace atm {



FluidModule::FluidModule()
    : m_current_fluid_task(0)
    , m_gravity_strength(15.0f)
{
    wdmAddNode("SPH/gravity_strength", &m_gravity_strength, 0.0f, 100.0f);
}

FluidModule::~FluidModule()
{
}

void FluidModule::initialize()
{
    m_rand.initialize(0);
}

void FluidModule::frameBegin()
{
    m_world.clearRigidsAndForces();
    m_current_fluid_task = 0;
}

void FluidModule::update( float32 dt )
{
    const psym::Particle *feedback = m_world.getParticles();
    uint32 n = m_world.getNumParticles();
    for(uint32 i=0; i<n; ++i) {
        const psym::Particle &m = feedback[i];
        if(m.hash==0) {
            if(IEntity *e = atmGetEntity(m.hit_to)) {
                atmCall(e, eventFluid, &m);
            }
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
        grav.strength = m_gravity_strength;
        m_world.addForce(grav);
    }
    atmGetCollisionModule()->copyRigitsToPSym();
}

void FluidModule::asyncupdate( float32 dt )
{
    m_mutex_particles.lock();
    m_particles_to_gpu.clear();

    ist::parallel_for(size_t(0), m_new_fluid_ctx.size(),
        [&](size_t i){
            AddFluidContext &ctx = m_new_fluid_ctx[i];
            const ParticleSet *rc           = atmGetParticleSet(ctx.psid);
            uint32 num_particles            = ctx.num;
            const PSetParticle *fluid_in    = rc->getParticleData();
            psym::Particle *fluid_out       = &m_new_fluid[ctx.index];

            simdvec4 zero(vec4(0.0f, 0.0f, 0.0f, 0.0f));
            simdmat4 t(ctx.mat);
            for(uint32 i=0; i<num_particles; ++i) {
                simdvec4 p(vec4(fluid_in[i].position, 1.0f));
                istAlign(16) vec4 pos = glm::vec4_cast(t * p);
                pos.z = stl::max<float32>(pos.z, 0.0f);
                fluid_out[i].position = reinterpret_cast<const psym::simdvec4&>(pos);
                fluid_out[i].velocity = reinterpret_cast<const psym::simdvec4&>(zero);
            }
        });
    if(!m_new_fluid.empty()) {
        addFluid(&m_new_fluid[0], m_new_fluid.size());
    }
    m_new_fluid_ctx.clear();
    m_new_fluid.clear();

    m_particles_to_gpu.insert(m_particles_to_gpu.end(), m_world.getParticles(), m_world.getParticles()+m_world.getNumParticles());
    m_mutex_particles.unlock();

    m_world.update(dt);
}

void FluidModule::draw()
{

}

void FluidModule::frameEnd()
{
}

size_t FluidModule::copyParticlesToGL()
{
    if(m_particles_to_gpu.empty()) { return 0; }

    ist::ScopedLock<ist::Mutex> l(m_mutex_particles);
    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    Buffer *vb = atmGetVertexBuffer(VBO_GB_FLUID);
    MapAndWrite(dc, vb, &m_particles_to_gpu[0], m_particles_to_gpu.size()*sizeof(psym::Particle));
    return m_particles_to_gpu.size();
}

size_t FluidModule::getNumParticles() const
{
    return m_world.getNumParticles();
}

void FluidModule::addRigid(const CollisionEntity &v)
{
    switch(v.getShapeType()) {
    case CS_Sphere:
        {
            const CollisionSphere &src = static_cast<const CollisionSphere&>(v);
            psym::RigidSphere dst;
            dst.id = src.getEntityHandle();
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

    case CS_Plane:
        {
            const CollisionPlane &src = static_cast<const CollisionPlane&>(v);
            psym::RigidPlane dst;
            dst.id = v.getEntityHandle();
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

    case CS_Box:
        {
            const CollisionBox &src = static_cast<const CollisionBox&>(v);
            psym::RigidBox dst;
            dst.id = v.getEntityHandle();
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
        istPrint("unknown type: 0x%p : %x", &v, v.getShapeType());
        istAssert(false);
    }
}

void FluidModule::addForce( const psym::PointForce &v )
{
    m_world.addForce(v);
}

void FluidModule::addFluid( psym::Particle *particles, uint32 num )
{
    const float32 energy_base = 2400.0f;
    const float32 energy_diffuse = 600.0f;
    for(uint32 i=0; i<num; ++i) {
        particles[i].energy = energy_base + (m_rand.genFloat32()*energy_diffuse);
        particles[i].density = 0.0f;
        particles[i].hash = 0;
        particles[i].hit_to = 0;
    }
    m_world.addParticles(particles, num);
}

void FluidModule::addFluid(PSET_RID psid, const mat4 &t, uint32 num)
{
    const ParticleSet *pset = atmGetParticleSet(psid);
    AddFluidContext ctx;
    ctx.psid = psid;
    ctx.mat = t;
    ctx.index = m_new_fluid.size();
    ctx.num = num==0 ? pset->getNumParticles() : std::min<uint32>(num, pset->getNumParticles());
    m_new_fluid.resize(m_new_fluid.size()+ctx.num);
    m_new_fluid_ctx.push_back(ctx);
}

void FluidModule::handleStateQuery( EntitiesQueryContext &ctx )
{
    const psym::Particle *particles = m_world.getParticles();
    size_t num = m_world.getNumParticles();
    size_t step = std::max<size_t>(num/100, 1);
    for(size_t i=0; i<num; i+=step) {
        vec2 pos = *((const vec2*)&particles[i].position);
        ctx.fluids.push_back(pos);
    }
}

} // namespace atm
