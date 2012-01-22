#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atomic {



class UpdateBloodstainParticle : public AtomicDrawTask
{
private:
    mat4 m_transform;
    const BloodstainParticle *m_bsp_in;
    BloodstainParticle *m_bsp_out;
    uint32 m_num_bsp;

public:
    void setup(const mat4 &transform, const BloodstainParticle *bsp_in, uint32 num_bsp, BloodstainParticle *bsp_out)
    {
        m_transform = transform;
        m_bsp_in    = bsp_in;
        m_num_bsp   = num_bsp;
        m_bsp_out   = bsp_out;
    }

    void exec()
    {
        uint32 num_particles            = m_num_bsp;
        const BloodstainParticle *bsp   = m_bsp_in;
        simdmat4 t(m_transform);
        for(uint32 i=0; i<num_particles; ++i) {
            simdvec4 p(bsp[i].position);
            m_bsp_out[i].position = glm::vec4_cast(t * p);
            m_bsp_out[i].params   = bsp[i].params;
        }
    }
};

PassDeferredShading_Bloodstain::PassDeferredShading_Bloodstain()
{
    m_ibo_sphere    = atomicGetIndexBuffer(IBO_BLOODSTAIN_SPHERE);
    m_va_sphere     = atomicGetVertexArray(VA_BLOOSTAIN_SPHERE);
    m_sh            = atomicGetShader(SH_BLOODSTAIN);
    m_vbo_bloodstain= atomicGetVertexBuffer(VBO_BLOODSTAIN_PARTICLES);
}

PassDeferredShading_Bloodstain::~PassDeferredShading_Bloodstain()
{
    for(uint32 i=0; i<m_tasks.size(); ++i) {
        istDelete(m_tasks[i]);
    }
    m_tasks.clear();
}

void PassDeferredShading_Bloodstain::beforeDraw()
{
    m_instances.clear();
    m_particles.clear();
}

void PassDeferredShading_Bloodstain::draw()
{
    if(m_instances.empty()) { return; }

    uint32 num_instances = m_instances.size();
    uint32 num_particles = 0;
    for(uint32 i=0; i<num_instances; ++i) {
        num_particles += m_instances[i].num_bsp;
    }

    m_particles.resize(num_particles);
    resizeTasks(num_instances);
    {
        uint32 n = 0;
        for(uint32 i=0; i<m_instances.size(); ++i) {
            const BloodstainParticleSet &bps = m_instances[i];
            static_cast<UpdateBloodstainParticle*>(m_tasks[i])->setup(
                bps.transform, bps.bsp_in, bps.num_bsp, &m_particles[n]);
            n += bps.num_bsp;
        }
    }
    TaskScheduler::addTask(&m_tasks[0], num_instances);
    TaskScheduler::waitFor(&m_tasks[0], num_instances);

    MapAndWrite(*m_vbo_bloodstain, &m_particles[0], sizeof(BloodstainParticle)*num_particles);


    RenderTarget *grt = atomicGetRenderTarget(RT_GENERIC);
    RenderTarget *gbuffer = atomicGetRenderTarget(RT_GBUFFER);
    grt->setColorBuffer(0, gbuffer->getColorBuffer(GBUFFER_COLOR));
    grt->setDepthStencilBuffer(gbuffer->getDepthStencilBuffer());
    grt->bind();
    gbuffer->getColorBuffer(GBUFFER_NORMAL)->bind(GLSL_NORMAL_BUFFER);
    gbuffer->getColorBuffer(GBUFFER_POSITION)->bind(GLSL_POSITION_BUFFER);

    glDisable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const VertexDesc descs[] = {
        {GLSL_INSTANCE_POSITION, I3D_FLOAT,4,  0, false, 1},
        {GLSL_INSTANCE_PARAM,    I3D_FLOAT,4, 16, false, 1},
    };
    m_sh->bind();
    m_va_sphere->bind();
    m_va_sphere->setAttributes(*m_vbo_bloodstain, sizeof(BloodstainParticle), descs, _countof(descs));
    m_ibo_sphere->bind();
    glDrawElementsInstanced(GL_QUADS, (8-1)*(8)*4, GL_UNSIGNED_INT, 0, num_particles);
    m_ibo_sphere->unbind();
    m_va_sphere->unbind();
    m_sh->unbind();

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);

    grt->unbind();
    atomicGetFrontRenderTarget()->bind();
}

void PassDeferredShading_Bloodstain::resizeTasks( uint32 n )
{
    while(m_tasks.size() < n) {
        m_tasks.push_back( istNew(UpdateBloodstainParticle)() );
    }
}

void PassDeferredShading_Bloodstain::addBloodstainParticles( const mat4 &t, const BloodstainParticle *bsp, uint32 num_bsp )
{
    if(num_bsp==0) { return; }

    BloodstainParticleSet tmp;
    tmp.transform   = t;
    tmp.bsp_in      = bsp;
    tmp.num_bsp     = num_bsp;
    m_instances.push_back(tmp);
}



PassDeferredShading_DirectionalLights::PassDeferredShading_DirectionalLights()
{
    m_instances.reserve(ATOMIC_MAX_DIRECTIONAL_LIGHTS);
}

void PassDeferredShading_DirectionalLights::beforeDraw()
{
    m_instances.clear();
}

void PassDeferredShading_DirectionalLights::draw()
{
    AtomicShader *shader    = atomicGetShader(SH_DIRECTIONALLIGHT);
    VertexArray *va_quad    = atomicGetVertexArray(VA_SCREEN_QUAD);
    Buffer *vbo_instance    = atomicGetVertexBuffer(VBO_DIRECTIONALLIGHT_INSTANCES);

    const uint32 num_instances = m_instances.size();
    MapAndWrite(*vbo_instance, &m_instances[0], sizeof(light_t)*num_instances);

    const VertexDesc descs[] = {
        {GLSL_INSTANCE_DIRECTION,I3D_FLOAT,4,  0, false, 1},
        {GLSL_INSTANCE_COLOR,    I3D_FLOAT,4, 16, false, 1},
        {GLSL_INSTANCE_AMBIENT,  I3D_FLOAT,4, 32, false, 1},
    };
    shader->bind();
    va_quad->bind();
    va_quad->setAttributes(*vbo_instance, sizeof(DirectionalLight), descs, _countof(descs));
    glDrawArraysInstanced(GL_QUADS, 0, 4, num_instances);
    va_quad->unbind();
    shader->unbind();
}

void PassDeferredShading_DirectionalLights::addInstance( const DirectionalLight& v )
{
    m_instances.push_back(v);
}



PassDeferredShading_PointLights::PassDeferredShading_PointLights()
{
    m_instances.reserve(1024);
}

void PassDeferredShading_PointLights::beforeDraw()
{
    m_instances.clear();
}

void PassDeferredShading_PointLights::draw()
{
    AtomicShader *shader    = atomicGetShader(SH_POINTLIGHT);
    Buffer *ibo_sphere      = atomicGetIndexBuffer(IBO_LIGHT_SPHERE);
    VertexArray *va_sphere  = atomicGetVertexArray(VA_UNIT_SPHERE);
    Buffer *vbo_instance    = atomicGetVertexBuffer(VBO_POINTLIGHT_INSTANCES);

    const uint32 num_instances = m_instances.size();
    MapAndWrite(*vbo_instance, &m_instances[0], sizeof(PointLight)*num_instances);

    const VertexDesc descs[] = {
        {GLSL_INSTANCE_POSITION,I3D_FLOAT,4, 0, false, 1},
        {GLSL_INSTANCE_COLOR,   I3D_FLOAT,4,16, false, 1},
        {GLSL_INSTANCE_PARAM,   I3D_FLOAT,4,32, false, 1},
    };

    shader->bind();
    va_sphere->bind();
    va_sphere->setAttributes(*vbo_instance, sizeof(PointLight), descs, _countof(descs));
    ibo_sphere->bind();
    glDrawElementsInstanced(GL_QUADS, (16-1)*(32)*4, GL_UNSIGNED_INT, 0, num_instances);
    ibo_sphere->unbind();
    va_sphere->unbind();
    shader->unbind();
}

} // namespace atomic
