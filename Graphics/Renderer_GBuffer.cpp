#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/SPHManager.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atomic {


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

    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    VertexArray     *va_cube  = atomicGetVertexArray(VA_FLUID_CUBE);
    Buffer          *vbo      = atomicGetVertexBuffer(VBO_PARTICLES);
    AtomicShader    *sh       = atomicGetShader(SH_GBUFFER_PARTICLES);

    MapAndWrite(dc, vbo, &m_particles[0], sizeof(IndivisualParticle)*m_particles.size());
    {
        const VertexDesc descs[] = {
            {GLSL_INSTANCE_POSITION, I3D_FLOAT32,4,  0, false, 1},
            {GLSL_INSTANCE_COLOR,    I3D_FLOAT32,4, 16, false, 1},
            {GLSL_INSTANCE_GLOW,     I3D_FLOAT32,4, 32, false, 1},
            {GLSL_INSTANCE_PARAM,    I3D_FLOAT32,4, 48, false, 1},
        };
        va_cube->setAttributes(1, vbo, 0, sizeof(IndivisualParticle), descs, _countof(descs));

        dc->setVertexArray(va_cube);
        sh->assign(dc);
        dc->drawInstanced(I3D_QUADS, 0, 24, m_particles.size());
    }
}

void PassGBuffer_Particle::addParticle( const IndivisualParticle *particles, uint32 num )
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
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    VertexArray     *va_cube  = atomicGetVertexArray(VA_FLUID_CUBE);
    Buffer          *vbo_fluid= atomicGetVertexBuffer(VBO_FLUID_PARTICLES);
    Buffer          *vbo_rigid= atomicGetVertexBuffer(VBO_RIGID_PARTICLES);
    AtomicShader    *sh_fluid = atomicGetShader(SH_GBUFFER_FLUID);
    AtomicShader    *sh_rigid = atomicGetShader(SH_GBUFFER_RIGID);

    // update rigid particle
    uint32 num_rigid_particles = 0;
    uint32 num_tasks = 0;
    {
        // 合計パーティクル数を算出して、それが収まるバッファを確保
        uint32 num_rigids = m_rupdateinfo.size();
        for(uint32 ri=0; ri<num_rigids; ++ri) {
            num_rigid_particles += atomicGetParticleSet(m_rupdateinfo[ri].psid)->getNumParticles();
        }
        m_rparticles.resize(num_rigid_particles);

        size_t n = 0;
        for(uint32 ri=0; ri<num_rigids; ++ri) {
            const ParticleSet *rc = atomicGetParticleSet(m_rupdateinfo[ri].psid);
            uint32 num_particles            = rc->getNumParticles();
            const PSetParticle *particles   = rc->getParticleData();
            for(uint32 i=0; i<num_particles; ++i) {
                uint32 pi = n+i;
                m_rparticles[pi].position     = particles[i].position;
                m_rparticles[pi].normal       = particles[i].normal;
                m_rparticles[pi].instanceid   = m_rupdateinfo[ri].instanceid;
            }
            n += atomicGetParticleSet(m_rupdateinfo[ri].psid)->getNumParticles();
        }
    }


    // fluid particle
    {
        // copy fluid particles (ispc -> GL)
        const uint32 num_particles = atomicGetSPHManager()->copyParticlesToGL();
        if(num_particles > 0) {
            const VertexDesc descs[] = {
                {GLSL_INSTANCE_POSITION, I3D_FLOAT32,4,  0, false, 1},
                {GLSL_INSTANCE_VELOCITY, I3D_FLOAT32,4, 16, false, 1},
                {GLSL_INSTANCE_PARAM,    I3D_FLOAT32,4, 32, false, 1},
            };
            va_cube->setAttributes(1, vbo_fluid, 0, sizeof(psym::Particle), descs, _countof(descs));
            sh_fluid->assign(dc);
            dc->setVertexArray(va_cube);
            dc->setDepthStencilState(atomicGetDepthStencilState(DS_GBUFFER_FLUID));
            dc->drawInstanced(I3D_QUADS, 0, 24, num_particles);
            dc->setVertexArray(NULL);
        }
    }

    // rigid particle
    Texture2D *param_texture = atomicGetTexture2D(TEX2D_ENTITY_PARAMS);
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
        dc->setDepthStencilState(atomicGetDepthStencilState(DS_GBUFFER_RIGID));
        dc->drawInstanced(I3D_QUADS, 0, 24, num_rigid_particles);
        dc->setDepthStencilState(atomicGetDepthStencilState(DS_GBUFFER_BG));
        dc->setVertexArray(NULL);
        dc->setTexture(GLSL_PARAM_BUFFER, NULL);
    }

    //// floor
    //{
    //    AtomicShader *sh_floor = atomicGetShader(SH_GBUFFER_FLOOR);
    //    VertexArray *va_floor = atomicGetVertexArray(VA_FLOOR_QUAD);
    //    sh_floor->assign(dc);
    //    dc->setVertexArray(va_floor);
    //    dc->draw(I3D_QUADS, 0, 4);
    //}
}

void PassGBuffer_Fluid::addPSetInstance( PSET_RID psid, const PSetInstance &inst )
{
    {
        const ParticleSet *rc = atomicGetParticleSet(psid);
        vec4 posf = inst.translate[3];
        posf.w = 0.0f;
        simdvec4 pos = simdvec4(posf);
        AABB aabb = rc->getAABB();
        aabb[0] = (simdvec4(aabb[0])+pos).Data;
        aabb[1] = (simdvec4(aabb[1])+pos).Data;
        if(!ist::TestFrustumAABB(*atomicGetViewFrustum(), aabb)) {
            return;
        }
    }

    PSetUpdateInfo tmp;
    tmp.psid        = psid;
    tmp.instanceid  = m_rinstances.size();
    m_rupdateinfo.push_back(tmp);
    m_rinstances.push_back(inst);
}





PassGBuffer_BG::PassGBuffer_BG()
    : m_enabled(true)
{
    atomicDbgAddParamNodeP("Rendering/BG/Enable", bool, &m_enabled);
}

PassGBuffer_BG::~PassGBuffer_BG()
{
    atomicDbgDeleteParamNode("Rendering/BG");
}

void PassGBuffer_BG::beforeDraw()
{
}

void PassGBuffer_BG::draw()
{
    return;
    if(!m_enabled) { return; }

    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    AtomicShader *sh_bg     = atomicGetShader(SH_BG1);
    AtomicShader *sh_up     = atomicGetShader(SH_GBUFFER_UPSAMPLING);
    VertexArray *va_quad    = atomicGetVertexArray(VA_SCREEN_QUAD);
    RenderTarget *gbuffer   = atomicGetRenderTarget(RT_GBUFFER);

    Buffer *ubo_rs          = atomicGetUniformBuffer(UBO_RENDERSTATES_3D);
    RenderStates *rs        = atomicGetRenderStates();


    if(atomicGetConfig()->bg_multiresolution) {
        // 1/4 の解像度で raymarching
        rs->ScreenSize      = vec2(atomicGetWindowSize())/4.0f;
        rs->RcpScreenSize   = vec2(1.0f, 1.0f) / rs->ScreenSize;
        MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));

        dc->setViewport(Viewport(ivec2(), gbuffer->getColorBuffer(0)->getDesc().size/4U));
        dc->setRenderTarget(NULL);
        dc->generateMips(gbuffer->getDepthStencilBuffer());
        gbuffer->setMipmapLevel(2);
        //dc->clearDepthStencil(gbuffer, 1.0f, 0);
        dc->setRenderTarget(gbuffer);

        sh_bg->bind();
        dc->setVertexArray(va_quad);
        dc->setDepthStencilState(atomicGetDepthStencilState(DS_GBUFFER_BG));
        dc->draw(I3D_QUADS, 0, 4);
        sh_bg->unbind();

        dc->setRenderTarget(NULL);
        gbuffer->setMipmapLevel(0);
        dc->setRenderTarget(gbuffer);
        dc->setViewport(Viewport(ivec2(), gbuffer->getColorBuffer(0)->getDesc().size));

        rs->ScreenSize      = vec2(atomicGetWindowSize());
        rs->RcpScreenSize   = vec2(1.0f, 1.0f) / rs->ScreenSize;
        MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));


        // 変化量少ない部分を upsampling
        dc->setTexture(GLSL_COLOR_BUFFER, gbuffer->getColorBuffer(GBUFFER_COLOR));
        dc->setTexture(GLSL_NORMAL_BUFFER, gbuffer->getColorBuffer(GBUFFER_NORMAL));
        dc->setTexture(GLSL_POSITION_BUFFER, gbuffer->getColorBuffer(GBUFFER_POSITION));
        dc->setTexture(GLSL_GLOW_BUFFER, gbuffer->getColorBuffer(GBUFFER_GLOW));
        dc->setVertexArray(va_quad);
        sh_up->bind();
        dc->setDepthStencilState(atomicGetDepthStencilState(DS_GBUFFER_UPSAMPLING));
        dc->draw(I3D_QUADS, 0, 4);
        sh_up->unbind();
        dc->setTexture(GLSL_COLOR_BUFFER, NULL);
        dc->setTexture(GLSL_NORMAL_BUFFER, NULL);
        dc->setTexture(GLSL_POSITION_BUFFER, NULL);
        dc->setTexture(GLSL_GLOW_BUFFER, NULL);
    }

    {
        sh_bg->bind();
        dc->setVertexArray(va_quad);
        dc->setDepthStencilState(atomicGetDepthStencilState(DS_GBUFFER_BG));
        dc->draw(I3D_QUADS, 0, 4);
        sh_bg->unbind();
    }
}

} // namespace atomic
