#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "GPGPU/SPH.cuh"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atomic {


UpdateRigidParticle::UpdateRigidParticle(const PSetUpdateInfo &ri, PSetParticle *p)
{
    m_rinst = &ri;
    m_particles = p;
}

void UpdateRigidParticle::exec()
{
    const ParticleSet *rc = atomicGetParticleSet(m_rinst->psid);
    uint32 num_particles            = rc->getNumParticles();
    const PSetParticle *particles   = rc->getParticleData();
    simdmat4 t(m_rinst->transform);
    for(uint32 i=0; i<num_particles; ++i) {
        simdvec4 p(vec4(particles[i].position, 1.0f));
        simdvec4 n((vec4&)particles[i].normal);
        m_particles[i].position     = vec3(glm::vec4_cast(t * p));
        m_particles[i].normal       = glm::vec4_cast(t * n);
        m_particles[i].instanceid   = m_rinst->instanceid;
    }
}

class UpdateRigidParticleTask : public AtomicDrawTask
{
private:
    UpdateRigidParticle *m_begin, *m_end;

public:
    void setup(UpdateRigidParticle *begin, UpdateRigidParticle *end) { m_begin=begin; m_end=end; }
    void exec()
    {
        for(UpdateRigidParticle *i=m_begin; i!=m_end; ++i) {
            i->exec();
        }
    }
};



PassGBuffer_SPH::PassGBuffer_SPH()
{
    m_va_cube       = atomicGetVertexArray(VA_FLUID_CUBE);
    m_sh_fluid      = atomicGetShader(SH_GBUFFER_FLUID);
    m_sh_rigid      = atomicGetShader(SH_GBUFFER_RIGID);
    m_vbo_fluid     = atomicGetVertexBuffer(VBO_FLUID_PARTICLES);
    m_vbo_rigid     = atomicGetVertexBuffer(VBO_RIGID_PARTICLES);
}

PassGBuffer_SPH::~PassGBuffer_SPH()
{
    for(uint32 i=0; i<m_tasks.size(); ++i) {
        istDelete(m_tasks[i]);
    }
    m_tasks.clear();
}

void PassGBuffer_SPH::beforeDraw()
{
    m_rupdateinfo.clear();
    m_rparticles.clear();
    m_rinstances.clear();
}

void PassGBuffer_SPH::draw()
{
    // update rigid particle
    m_updater.clear();
    uint32 num_rigid_particles = 0;
    uint32 num_tasks = 0;
    {
        // 合計パーティクル数を算出して、それが収まるバッファを確保
        uint32 num_rigids = m_rupdateinfo.size();
        for(uint32 ri=0; ri<num_rigids; ++ri) {
            num_rigid_particles += atomicGetParticleSet(m_rupdateinfo[ri].psid)->getNumParticles();
        }
        m_rparticles.resize(num_rigid_particles);

        // 並列更新に必要な情報を設定
        size_t n = 0;
        for(uint32 ri=0; ri<num_rigids; ++ri) {
            m_updater.push_back( UpdateRigidParticle(m_rupdateinfo[ri], &m_rparticles[n]) );
            n += atomicGetParticleSet(m_rupdateinfo[ri].psid)->getNumParticles();
        }

        // 並列更新の粒度を設定 (一定頂点数でタスクを分割)
        const uint32 minimum_particles_in_task = 5000;
        UpdateRigidParticle *last = &m_updater[0];
        uint32 particles_in_task = 0;
        for(uint32 ri=0; ri<num_rigids; ++ri) {
            particles_in_task += atomicGetParticleSet(m_rupdateinfo[ri].psid)->getNumParticles();
            if(particles_in_task > minimum_particles_in_task || ri+1==num_rigids) {
                UpdateRigidParticle *current = &m_updater[0]+(ri+1);
                resizeTasks(num_tasks+1);
                static_cast<UpdateRigidParticleTask*>(m_tasks[num_tasks])->setup(last, current);
                last = current;
                ++num_tasks;
                particles_in_task = 0;
            }
        }

        // 並列頂点更新開始
        TaskScheduler::addTask(&m_tasks[0], num_tasks);
    }
    // copy fluid particles (CUDA -> GL)
    atomicGetSPHManager()->copyParticlesToGL();
    const sphStates& sphs = SPHGetStates();


    // fluid particle
    {
        const uint32 num_particles = sphs.fluid_num_particles;
        const VertexDesc descs[] = {
            {GLSL_INSTANCE_PARAM,    I3D_FLOAT,4,  0, false, 1},
            {GLSL_INSTANCE_POSITION, I3D_FLOAT,4, 16, false, 1},
            {GLSL_INSTANCE_VELOCITY, I3D_FLOAT,4, 32, false, 1},
        };
        m_sh_fluid->bind();
        m_va_cube->bind();
        m_va_cube->setAttributes(*m_vbo_fluid, sizeof(sphFluidParticle), descs, _countof(descs));
        glDrawArraysInstanced(GL_QUADS, 0, 24, num_particles);
        m_va_cube->unbind();
        m_sh_fluid->unbind();
    }

    // rigid particle
    Texture2D *param_texture = atomicGetTexture2D(TEX2D_ENTITY_PARAMS);
    {
        param_texture->copy(0, uvec2(0,0), uvec2(sizeof(PSetInstance)/sizeof(vec4), m_rinstances.size()), I3D_RGBA32F, &m_rinstances[0]);
        TaskScheduler::waitFor(&m_tasks[0], num_tasks);
        MapAndWrite(*m_vbo_rigid, &m_rparticles[0], sizeof(PSetParticle)*num_rigid_particles);
    }
    {
        const VertexDesc descs[] = {
            {GLSL_INSTANCE_NORMAL,   I3D_FLOAT,4,  0, false, 1},
            {GLSL_INSTANCE_POSITION, I3D_FLOAT,3, 16, false, 1},
            {GLSL_INSTANCE_PARAM,    I3D_INT,  1, 28, false, 1},
        };
        m_sh_rigid->bind();
        m_va_cube->bind();
        m_va_cube->setAttributes(*m_vbo_rigid, sizeof(PSetParticle), descs, _countof(descs));
        param_texture->bind(GLSL_PARAM_BUFFER);
        glDrawArraysInstanced(GL_QUADS, 0, 24, num_rigid_particles);
        m_va_cube->unbind();
        m_sh_rigid->unbind();
    }

    // floor
    {
        AtomicShader *sh_floor = atomicGetShader(SH_GBUFFER_FLOOR);
        VertexArray *va_floor = atomicGetVertexArray(VA_FLOOR_QUAD);
        sh_floor->bind();
        va_floor->bind();
        glDrawArrays(GL_QUADS, 0, 4);
        va_floor->unbind();
        sh_floor->unbind();
    }
}

void PassGBuffer_SPH::resizeTasks( uint32 n )
{
    while(m_tasks.size() < n) {
        m_tasks.push_back( istNew(UpdateRigidParticleTask)() );
    }
}

void PassGBuffer_SPH::addPSetInstance( PSET_RID psid, const mat4 &t, const PSetInstance inst )
{
    // todo: AABB 適切に設定
    {
        simdvec4 pos = simdvec4(t[3]);
        AABB aabb = AABB(pos.Data);
        if(!ist::TestFrustumAABB(*atomicGetViewFrustum(), aabb)) {
            return;
        }
    }

    PSetUpdateInfo tmp;
    tmp.psid        = psid;
    tmp.instanceid  = m_rinstances.size();
    tmp.transform   = t;
    m_rupdateinfo.push_back(tmp);
    m_rinstances.push_back(inst);
}




} // namespace atomic
