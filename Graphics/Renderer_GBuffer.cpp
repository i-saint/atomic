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
    m_rigid_sp.vbo    = VBO_GB_RIGID_SPHERICAL;
    m_rigid_sp.shader = SH_GBUFFER_RIGID_SPHERICAL;
    m_rigid_sp.params = TEX2D_PSET_PARAMS_GB_SP;

    m_rigid_so.vbo    = VBO_GB_RIGID_SOLID;
    m_rigid_so.shader = SH_GBUFFER_RIGID_SOLID;
    m_rigid_so.params = TEX2D_PSET_PARAMS_GB_SO;
}

PassGBuffer_Fluid::~PassGBuffer_Fluid()
{
}

void PassGBuffer_Fluid::beforeDraw()
{
    m_rigid_sp.clear();
    m_rigid_so.clear();
}

void PassGBuffer_Fluid::draw()
{
    drawFluid();
    drawParticleSets(m_rigid_sp);
    drawParticleSets(m_rigid_so);
}

void PassGBuffer_Fluid::drawFluid()
{
    i3d::DeviceContext  *dc = atmGetGLDeviceContext();
    VertexArray         *va_cube  = atmGetVertexArray(VA_FLUID_CUBE);
    Buffer              *vbo_fluid= atmGetVertexBuffer(VBO_GB_FLUID);
    AtomicShader        *sh_fluid = atmGetShader(SH_GBUFFER_FLUID_SPHERICAL);

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
            dc->setVertexArray(nullptr);
        }
    }
}

void PassGBuffer_Fluid::drawParticleSets( PSetDrawData &pdd )
{
    if(pdd.update_info.empty()) { return; }

    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    VertexArray     *va_cube  = atmGetVertexArray(VA_FLUID_CUBE);
    Buffer          *vbo = atmGetVertexBuffer(pdd.vbo);
    AtomicShader    *shader = atmGetShader(pdd.shader);
    Texture2D       *params = atmGetTexture2D(pdd.params);

    // update rigid particle
    uint32 num_particles = 0;
    uint32 num_tasks = 0;
    {
        // 合計パーティクル数を算出して、それが収まるバッファを確保
        uint32 num_instances = pdd.update_info.size();
        for(uint32 ri=0; ri<num_instances; ++ri) {
            num_particles += atmGetParticleSet(pdd.update_info[ri].psid)->getNumParticles();
        }
        pdd.particle_data.resize(num_particles);

        size_t n = 0;
        for(uint32 ri=0; ri<num_instances; ++ri) {
            const ParticleSet *rc = atmGetParticleSet(pdd.update_info[ri].psid);
            uint32 num_particles            = rc->getNumParticles();
            const PSetParticle *particles   = rc->getParticleData();
            for(uint32 i=0; i<num_particles; ++i) {
                uint32 pi = n+i;
                pdd.particle_data[pi].position     = particles[i].position;
                pdd.particle_data[pi].normal       = particles[i].normal;
                pdd.particle_data[pi].instanceid   = pdd.update_info[ri].instanceid;
            }
            n += atmGetParticleSet(pdd.update_info[ri].psid)->getNumParticles();
        }
    }

    if(!pdd.instance_data.empty()) {
        dc->updateResource(params, 0, uvec2(0,0), uvec2(sizeof(PSetInstance)/sizeof(vec4), pdd.instance_data.size()), &pdd.instance_data[0]);
        MapAndWrite(dc, vbo, &pdd.particle_data[0], sizeof(PSetParticle)*num_particles);
    }
    {
        const VertexDesc descs[] = {
            {GLSL_INSTANCE_NORMAL,   I3D_FLOAT32,4,  0, false, 1},
            {GLSL_INSTANCE_POSITION, I3D_FLOAT32,3, 16, false, 1},
            {GLSL_INSTANCE_PARAM,    I3D_INT32,  1, 28, false, 1},
        };
        va_cube->setAttributes(1, vbo, 0, sizeof(PSetParticle), descs, _countof(descs));

        shader->assign(dc);
        dc->setTexture(GLSL_PARAM_BUFFER, params);
        dc->setVertexArray(va_cube);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_RIGID));
        dc->drawInstanced(I3D_QUADS, 0, 24, num_particles);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_BG));
        dc->setVertexArray(nullptr);
        dc->setTexture(GLSL_PARAM_BUFFER, NULL);
    }
}

void PassGBuffer_Fluid::addParticles( PSET_RID psid, const PSetInstance &inst )
{
    if(!culling(psid, inst)) { return; }

    PSetUpdateInfo tmp;
    tmp.psid        = psid;
    tmp.instanceid  = m_rigid_sp.instance_data.size();
    m_rigid_sp.update_info.push_back(tmp);
    m_rigid_sp.instance_data.push_back(inst);
}

void PassGBuffer_Fluid::addParticlesSolid( PSET_RID psid, const PSetInstance &inst )
{
    if(!culling(psid, inst)) { return; }

    PSetUpdateInfo tmp;
    tmp.psid        = psid;
    tmp.instanceid  = m_rigid_so.instance_data.size();
    m_rigid_so.update_info.push_back(tmp);
    m_rigid_so.instance_data.push_back(inst);
}

bool PassGBuffer_Fluid::culling( PSET_RID psid, const PSetInstance &inst )
{
    const ParticleSet *rc = atmGetParticleSet(psid);
    vec4 posf = inst.transform[3];
    posf.w = 0.0f;
    simdvec4 pos = simdvec4(posf);
    AABB aabb = rc->getAABB();
    aabb[0] = (simdvec4(aabb[0])+pos).Data;
    aabb[1] = (simdvec4(aabb[1])+pos).Data;
    if(!ist::TestFrustumAABB(*atmGetViewFrustum(), aabb)) {
        return false;
    }
    return true;
}


} // namespace atm
