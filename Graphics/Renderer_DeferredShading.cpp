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
            assign_float4(m_bsp_out[i].params, bsp[i].params);
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

    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
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
    ist::EnqueueTasks(&m_tasks[0], num_instances);
    ist::WaitTasks(&m_tasks[0], num_instances);

    MapAndWrite(*m_vbo_bloodstain, &m_particles[0], sizeof(BloodstainParticle)*num_particles);

    // RT_GBUFFER のカラーバッファを更新する。
    // そのため、GLSL_COLOR_BUFFER は一時的に unbind

    RenderTarget *grt = atomicGetRenderTarget(RT_GENERIC);
    RenderTarget *gbuffer = atomicGetRenderTarget(RT_GBUFFER);
    dc->setTexture(GLSL_COLOR_BUFFER, NULL);
    grt->setColorBuffer(0, gbuffer->getColorBuffer(GBUFFER_COLOR));
    grt->setDepthStencilBuffer(gbuffer->getDepthStencilBuffer());

    glDisable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    const VertexDesc descs[] = {
        {GLSL_INSTANCE_POSITION, I3D_FLOAT,4,  0, false, 1},
        {GLSL_INSTANCE_PARAM,    I3D_FLOAT,4, 16, false, 1},
    };
    m_va_sphere->setAttributes(*m_vbo_bloodstain, sizeof(BloodstainParticle), descs, _countof(descs));

    dc->setRenderTarget(grt);
    m_sh->assign(dc);
    dc->setRenderTarget(grt);
    dc->setVertexArray(m_va_sphere);
    dc->setIndexBuffer(m_ibo_sphere, I3D_UINT);
    dc->drawIndexedInstanced(I3D_QUADS, 0, (8-1)*(8)*4, num_particles);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);

    dc->setRenderTarget(atomicGetFrontRenderTarget());
    dc->setTexture(GLSL_COLOR_BUFFER, gbuffer->getColorBuffer(GBUFFER_COLOR));
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



PassDeferredShading_Lights::PassDeferredShading_Lights()
    : m_rendered_lights(0)
{
    m_mr_params.Level = ivec4(0,0,0,0);
    m_mr_params.Threshold = vec4(0.95f, 0.0f, 0.0f, 0.0f);

    m_directional_lights.reserve(ATOMIC_MAX_DIRECTIONAL_LIGHTS);
    m_point_lights.reserve(ATOMIC_MAX_POINT_LIGHTS);
}

void PassDeferredShading_Lights::beforeDraw()
{
    m_directional_lights.clear();
    m_point_lights.clear();
    m_rendered_lights = 0;
}

void PassDeferredShading_Lights::draw()
{
    updateConstantBuffers();
    if(atomicGetConfig()->multiresolution) {
        drawMultiResolution();
    }
    else {
        drawLights();
    }
}

void PassDeferredShading_Lights::drawMultiResolution()
{
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    RenderTarget *rt_gbuffer    = atomicGetRenderTarget(RT_GBUFFER);
    RenderTarget *rt_quarter    = atomicGetRenderTarget(RT_OUTPUT_QUARTER);
    RenderTarget *rt_half       = atomicGetRenderTarget(RT_OUTPUT_HALF);
    RenderTarget *rt_original   = atomicGetFrontRenderTarget();

    dc->setSampler(GLSL_BACK_BUFFER, atomicGetSampler(SAMPLER_GBUFFER));

    // 1/4 の解像度で shading
    {
        Viewport vp(ivec2(), rt_quarter->getColorBuffer(0)->getDesc().size);
        dc->setViewport(vp);

        rt_quarter->setDepthStencilBuffer(rt_gbuffer->getDepthStencilBuffer(), 2);
        rt_quarter->bind();
        glClear(GL_COLOR_BUFFER_BIT);

        drawLights();
        debugShowResolution(2);
        rt_quarter->unbind();
    }

    // 1/2
    {
        Viewport vp(ivec2(), rt_half->getColorBuffer(0)->getDesc().size);
        dc->setViewport(vp);

        rt_half->setDepthStencilBuffer(rt_gbuffer->getDepthStencilBuffer(), 1);
        rt_half->bind();
        glClear(GL_COLOR_BUFFER_BIT);

        upsampling(2);
        drawLights();
        debugShowResolution(1);

        rt_half->unbind();
    }

    // 1/1
    {
        dc->setViewport(*atomicGetDefaultViewport());
        rt_original->bind();

        upsampling(1);
        drawLights();
        debugShowResolution(0);

        // not unbind
    }

    dc->setSampler(GLSL_BACK_BUFFER, atomicGetSampler(SAMPLER_TEXTURE_DEFAULT));
}

void PassDeferredShading_Lights::debugShowResolution( int32 level )
{
    static const vec4 colors[] = {
        vec4(0.8f, 0.0f, 0.0f, 0.7f),
        vec4(0.5f, 0.0f, 0.0f, 0.7f),
        vec4(0.2f, 0.0f, 0.0f, 0.7f),
    };
    if(atomicGetConfig()->debug_show_resolution) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        FillScreen(colors[level]);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }
}

void PassDeferredShading_Lights::upsampling(int32 level)
{
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    AtomicShader *sh_upsampling = atomicGetShader(SH_UPSAMPLING);
    VertexArray *va_quad        = atomicGetVertexArray(VA_SCREEN_QUAD);
    Buffer *ubo_mrp             = atomicGetUniformBuffer(UBO_MULTIRESOLUTION_PARAMS);
    int mr_params_loc = sh_upsampling->getUniformBlockIndex("multiresolution_params");

    Texture2D *lower_resolution = NULL;
    RenderTarget *rt = NULL;
    if(level==2) {
        lower_resolution = atomicGetRenderTarget(RT_OUTPUT_QUARTER)->getColorBuffer(0);
        rt = atomicGetRenderTarget(RT_OUTPUT_HALF);
    }
    else if(level==1) {
        lower_resolution = atomicGetRenderTarget(RT_OUTPUT_HALF)->getColorBuffer(0);
        rt = atomicGetFrontRenderTarget();
    }
    else {
        istPrint("PassDeferredShading_Lights::upsampling(): invalid level\n");
        return;
    }

    {
        m_mr_params.Level.x = level;
        MapAndWrite(*ubo_mrp, &m_mr_params, sizeof(m_mr_params));
    }
    
    glDepthFunc(GL_ALWAYS);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    sh_upsampling->bind();
    sh_upsampling->setUniformBlock(mr_params_loc, GLSL_MULTIRESOLUTION_BINDING, ubo_mrp->getHandle());
    lower_resolution->bind(GLSL_BACK_BUFFER);
    va_quad->bind();
    dc->draw(I3D_QUADS, 0, 4);
    va_quad->unbind();
    sh_upsampling->unbind();

    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LESS);
}

void PassDeferredShading_Lights::updateConstantBuffers()
{
    {
        Buffer *vbo_instance    = atomicGetVertexBuffer(VBO_DIRECTIONALLIGHT_INSTANCES);
        int32 num_lights = m_directional_lights.size();
        MapAndWrite(*vbo_instance, &m_directional_lights[0], sizeof(DirectionalLight)*num_lights);
    }
    {
        Buffer *vbo_instance    = atomicGetVertexBuffer(VBO_POINTLIGHT_INSTANCES);
        int32 num_lights = m_point_lights.size();
        MapAndWrite(*vbo_instance, &m_point_lights[0], sizeof(PointLight)*num_lights);
    }
}

void PassDeferredShading_Lights::drawLights()
{
    m_rendered_lights = 0;
    drawDirectionalLights();
    drawPointLights();
}

void PassDeferredShading_Lights::drawDirectionalLights()
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    AtomicShader *shader    = atomicGetShader(SH_DIRECTIONALLIGHT);
    VertexArray *va_quad    = atomicGetVertexArray(VA_SCREEN_QUAD);
    Buffer *vbo_instance    = atomicGetVertexBuffer(VBO_DIRECTIONALLIGHT_INSTANCES);

    int32 num_lights = m_directional_lights.size();
    int32 show = atomicGetConfig()->debug_show_lights - m_rendered_lights;
    if(show >= 0) {
        num_lights = stl::min(num_lights, show);
    }
    m_rendered_lights += num_lights;

    const VertexDesc descs[] = {
        {GLSL_INSTANCE_DIRECTION,I3D_FLOAT,4,  0, false, 1},
        {GLSL_INSTANCE_COLOR,    I3D_FLOAT,4, 16, false, 1},
        {GLSL_INSTANCE_AMBIENT,  I3D_FLOAT,4, 32, false, 1},
    };
    va_quad->setAttributes(*vbo_instance, sizeof(DirectionalLight), descs, _countof(descs));

    shader->assign(dc);
    dc->setVertexArray(va_quad);
    dc->drawInstanced(I3D_QUADS, 0, 4, num_lights);
}

void PassDeferredShading_Lights::drawPointLights()
{
    i3d::DeviceContext *dc  = atomicGetGLDeviceContext();
    AtomicShader *shader    = atomicGetShader(SH_POINTLIGHT);
    Buffer *ibo_sphere      = atomicGetIndexBuffer(IBO_LIGHT_SPHERE);
    VertexArray *va_sphere  = atomicGetVertexArray(VA_UNIT_SPHERE);
    Buffer *vbo_instance    = atomicGetVertexBuffer(VBO_POINTLIGHT_INSTANCES);

    int32 num_lights = m_point_lights.size();
    int32 show = atomicGetConfig()->debug_show_lights - m_rendered_lights;
    if(show >= 0) {
        num_lights = stl::min(num_lights, show);
    }
    m_rendered_lights += num_lights;

    const VertexDesc descs[] = {
        {GLSL_INSTANCE_POSITION,I3D_FLOAT,4, 0, false, 1},
        {GLSL_INSTANCE_COLOR,   I3D_FLOAT,4,16, false, 1},
        {GLSL_INSTANCE_PARAM,   I3D_FLOAT,4,32, false, 1},
    };
    va_sphere->setAttributes(*vbo_instance, sizeof(PointLight), descs, _countof(descs));

    shader->assign(dc);
    dc->setVertexArray(va_sphere);
    dc->setIndexBuffer(ibo_sphere, I3D_UINT);
    dc->drawIndexedInstanced(I3D_QUADS, 0, (16-1)*(32)*4, num_lights);
}

void PassDeferredShading_Lights::addLight( const DirectionalLight& v )
{
    m_directional_lights.push_back(v);
}

void PassDeferredShading_Lights::addLight( const PointLight& v )
{
    m_point_lights.push_back(v);
}

} // namespace atomic
