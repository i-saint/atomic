#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/FluidModule.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atm {


PassGBuffer_Particle::PassGBuffer_Particle()
{
}

PassGBuffer_Particle::~PassGBuffer_Particle()
{
}

void PassGBuffer_Particle::beforeDraw()
{
    m_particles.clear();
}

void PassGBuffer_Particle::draw()
{
    if(m_particles.empty()) { return; }

    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    VertexArray     *va_cube  = atmGetVertexArray(VA_FLUID_CUBE);
    Buffer          *vbo      = atmGetVertexBuffer(VBO_PARTICLES);
    AtomicShader    *sh       = atmGetShader(SH_GBUFFER_PARTICLES);

    MapAndWrite(dc, vbo, &m_particles[0], sizeof(SingleParticle)*m_particles.size());
    {
        const VertexDesc descs[] = {
            {GLSL_INSTANCE_POSITION, I3D_FLOAT32,4,  0, false, 1},
            {GLSL_INSTANCE_COLOR,    I3D_FLOAT32,4, 16, false, 1},
            {GLSL_INSTANCE_GLOW,     I3D_FLOAT32,4, 32, false, 1},
            {GLSL_INSTANCE_PARAM,    I3D_FLOAT32,4, 48, false, 1},
        };
        va_cube->setAttributes(1, vbo, 0, sizeof(SingleParticle), descs, _countof(descs));

        dc->setVertexArray(va_cube);
        sh->assign(dc);
        dc->drawInstanced(I3D_QUADS, 0, 24, m_particles.size());
    }
}

void PassGBuffer_Particle::addParticle( const SingleParticle *particles, uint32 num )
{
    m_particles.insert(m_particles.end(), particles, particles+num);
}




PassGBuffer_Fluid::PassGBuffer_Fluid()
{
}

PassGBuffer_Fluid::~PassGBuffer_Fluid()
{
}

void PassGBuffer_Fluid::beforeDraw()
{
    m_rupdateinfo.clear();
    m_rparticles.clear();
    m_rinstances.clear();
}

void PassGBuffer_Fluid::draw()
{
    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    VertexArray     *va_cube  = atmGetVertexArray(VA_FLUID_CUBE);
    Buffer          *vbo_fluid= atmGetVertexBuffer(VBO_FLUID_PARTICLES);
    Buffer          *vbo_rigid= atmGetVertexBuffer(VBO_RIGID_PARTICLES);
    AtomicShader    *sh_fluid = atmGetShader(SH_GBUFFER_FLUID_SPHERICAL);
    //AtomicShader    *sh_rigid = atmGetShader(SH_GBUFFER_RIGID_SPHERICAL);
    AtomicShader    *sh_rigid = atmGetShader(SH_GBUFFER_RIGID_SOLID);

    // update rigid particle
    uint32 num_rigid_particles = 0;
    uint32 num_tasks = 0;
    {
        // 合計パーティクル数を算出して、それが収まるバッファを確保
        uint32 num_rigids = m_rupdateinfo.size();
        for(uint32 ri=0; ri<num_rigids; ++ri) {
            num_rigid_particles += atmGetParticleSet(m_rupdateinfo[ri].psid)->getNumParticles();
        }
        m_rparticles.resize(num_rigid_particles);

        size_t n = 0;
        for(uint32 ri=0; ri<num_rigids; ++ri) {
            const ParticleSet *rc = atmGetParticleSet(m_rupdateinfo[ri].psid);
            uint32 num_particles            = rc->getNumParticles();
            const PSetParticle *particles   = rc->getParticleData();
            for(uint32 i=0; i<num_particles; ++i) {
                uint32 pi = n+i;
                m_rparticles[pi].position     = particles[i].position;
                m_rparticles[pi].normal       = particles[i].normal;
                m_rparticles[pi].instanceid   = m_rupdateinfo[ri].instanceid;
            }
            n += atmGetParticleSet(m_rupdateinfo[ri].psid)->getNumParticles();
        }
    }


    // fluid particle
    if(atmGetGame()) {
        // copy fluid particles (ispc -> GL)
        const uint32 num_particles = atmGetFluidModule()->copyParticlesToGL();
        if(num_particles > 0) {
            const VertexDesc descs[] = {
                {GLSL_INSTANCE_POSITION, I3D_FLOAT32,4,  0, false, 1},
                {GLSL_INSTANCE_VELOCITY, I3D_FLOAT32,4, 16, false, 1},
                {GLSL_INSTANCE_PARAM,    I3D_FLOAT32,4, 32, false, 1},
            };
            va_cube->setAttributes(1, vbo_fluid, 0, sizeof(psym::Particle), descs, _countof(descs));
            sh_fluid->assign(dc);
            dc->setVertexArray(va_cube);
            dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_FLUID));
            dc->drawInstanced(I3D_QUADS, 0, 24, num_particles);
            dc->setVertexArray(NULL);
        }
    }

    // rigid particle
    Texture2D *param_texture = atmGetTexture2D(TEX2D_ENTITY_PARAMS_GBUFFER);
    if(!m_rinstances.empty()) {
        dc->updateResource(param_texture, 0, uvec2(0,0), uvec2(sizeof(PSetInstance)/sizeof(vec4), m_rinstances.size()), &m_rinstances[0]);
        MapAndWrite(dc, vbo_rigid, &m_rparticles[0], sizeof(PSetParticle)*num_rigid_particles);
    }
    {
        const VertexDesc descs[] = {
            {GLSL_INSTANCE_NORMAL,   I3D_FLOAT32,4,  0, false, 1},
            {GLSL_INSTANCE_POSITION, I3D_FLOAT32,3, 16, false, 1},
            {GLSL_INSTANCE_PARAM,    I3D_INT32,  1, 28, false, 1},
        };
        va_cube->setAttributes(1, vbo_rigid, 0, sizeof(PSetParticle), descs, _countof(descs));

        sh_rigid->assign(dc);
        dc->setTexture(GLSL_PARAM_BUFFER, param_texture);
        dc->setVertexArray(va_cube);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_RIGID));
        dc->drawInstanced(I3D_QUADS, 0, 24, num_rigid_particles);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_BG));
        dc->setVertexArray(NULL);
        dc->setTexture(GLSL_PARAM_BUFFER, NULL);
    }
}

void PassGBuffer_Fluid::addPSetInstance( PSET_RID psid, const PSetInstance &inst )
{
    {
        const ParticleSet *rc = atmGetParticleSet(psid);
        vec4 posf = inst.transform[3];
        posf.w = 0.0f;
        simdvec4 pos = simdvec4(posf);
        AABB aabb = rc->getAABB();
        aabb[0] = (simdvec4(aabb[0])+pos).Data;
        aabb[1] = (simdvec4(aabb[1])+pos).Data;
        if(!ist::TestFrustumAABB(*atmGetViewFrustum(), aabb)) {
            return;
        }
    }

    PSetUpdateInfo tmp;
    tmp.psid        = psid;
    tmp.instanceid  = m_rinstances.size();
    m_rupdateinfo.push_back(tmp);
    m_rinstances.push_back(inst);
}


} // namespace atm
