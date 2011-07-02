#include "stdafx.h"
#include "../types.h"
#include "../Game/AtomicApplication.h"
#include "../Game/AtomicGame.h"
#include "../Game/World.h"
#include "Renderer.h"

namespace atomic {


AtomicRenderer* AtomicRenderer::s_inst = NULL;

void AtomicRenderer::initializeInstance()
{
    if(!s_inst) {
        s_inst = AT_NEW(AtomicRenderer) AtomicRenderer();
    }
    else {
        IST_ASSERT("already initialized");
    }
}

void AtomicRenderer::finalizeInstance()
{
    AT_DELETE(s_inst);
}

AtomicRenderer::AtomicRenderer()
{
    m_sh_gbuffer    = GetShaderGBuffer();
    m_sh_deferred   = GetShaderDeferred();
    m_sh_out        = GetShaderOutput();

    m_rt_gbuffer    = GetRenderTargetGBuffer();
    m_rt_deferred   = GetRenderTargetDeferred();

    m_renderer_cube = AT_NEW(PassGBuffer_Cube) PassGBuffer_Cube();
    m_renderer_sphere_light = AT_NEW(PassDeferred_SphereLight) PassDeferred_SphereLight();
    m_renderers[PASS_GBUFFER].push_back(m_renderer_cube);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_sphere_light);
}

AtomicRenderer::~AtomicRenderer()
{
    AT_DELETE(m_renderer_sphere_light);
    AT_DELETE(m_renderer_cube);
}

void AtomicRenderer::beforeDraw()
{
    for(uint32 i=0; i<_countof(m_renderers); ++i) {
        uint32 size = m_renderers[i].size();
        for(uint32 j=0; j<size; ++j) {
            m_renderers[i][j]->beforeDraw();
        }
    }
}

void AtomicRenderer::draw()
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);

    glLoadIdentity();

    glEnable(GL_DEPTH_TEST);

    pass_Shadow();
    pass_GBuffer();
    pass_Deferred();
    pass_Forward();
    pass_Postprocess();
    pass_UI();
    pass_Output();

    glSwapBuffers();
}

void AtomicRenderer::pass_Shadow()
{
    glClear(GL_DEPTH_BUFFER_BIT);
    glFrontFace(GL_CW);

    uint32 num_renderers = m_renderers[PASS_SHADOW_DEPTH].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_SHADOW_DEPTH][i]->draw();
    }

    glFrontFace(GL_CCW);
}

void AtomicRenderer::pass_GBuffer()
{
    const PerspectiveCamera *camera = GetCamera();

    m_rt_gbuffer->bind();
    m_sh_gbuffer->bind();
    camera->bind();
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    uint32 num_renderers = m_renderers[PASS_GBUFFER].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_GBUFFER][i]->draw();
    }

    m_sh_gbuffer->unbind();
    m_rt_gbuffer->unbind();
}

void AtomicRenderer::pass_Deferred()
{
    const PerspectiveCamera *camera = GetCamera();
    float aspect_ratio = camera->getAspect();
    float x_scale = float32(GetWindowWidth())/float32(m_rt_deferred->getWidth());
    float y_scale = float32(GetWindowHeight())/float32(m_rt_deferred->getHeight()) * aspect_ratio;

    m_rt_deferred->bind();
    m_sh_deferred->bind();
    camera->bind();
    m_rt_gbuffer->getColorBuffer(0)->bind(Texture2D::SLOT_0);
    m_rt_gbuffer->getColorBuffer(1)->bind(Texture2D::SLOT_1);
    m_rt_gbuffer->getColorBuffer(2)->bind(Texture2D::SLOT_2);
    m_sh_deferred->setColorBuffer(Texture2D::SLOT_0);
    m_sh_deferred->setNormalBuffer(Texture2D::SLOT_1);
    m_sh_deferred->setPositionBuffer(Texture2D::SLOT_2);
    m_sh_deferred->setAspectRatio(aspect_ratio);
    m_sh_deferred->setTexcoordScale(x_scale, y_scale);
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);

    uint32 num_renderers = m_renderers[PASS_DEFERRED].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_DEFERRED][i]->draw();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    m_rt_gbuffer->getColorBuffer(2)->unbind(Texture2D::SLOT_2);
    m_rt_gbuffer->getColorBuffer(1)->unbind(Texture2D::SLOT_1);
    m_rt_gbuffer->getColorBuffer(0)->unbind(Texture2D::SLOT_0);
    m_sh_deferred->unbind();
    m_rt_deferred->unbind();
}

void AtomicRenderer::pass_Forward()
{
    uint32 num_renderers = m_renderers[PASS_FORWARD].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_FORWARD][i]->draw();
    }
}

void AtomicRenderer::pass_Postprocess()
{
    uint32 num_renderers = m_renderers[PASS_POSTPROCESS].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_POSTPROCESS][i]->draw();
    }
}

void AtomicRenderer::pass_UI()
{
    uint32 num_renderers = m_renderers[PASS_UI].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_UI][i]->draw();
    }
}

void AtomicRenderer::pass_Output()
{
    m_rt_deferred->getColorBuffer(0)->bind(Texture2D::SLOT_0);
    m_sh_out->bind();
    m_sh_out->setUniform1i(m_sh_out->getUniformLocation("ColorBuffer"), Texture2D::SLOT_0);
    DrawScreen(0.0f, 0.0f, float(GetWindowWidth())/float32(m_rt_deferred->getWidth()), float(GetWindowHeight())/float32(m_rt_deferred->getHeight()));
    m_sh_out->unbind();
    m_rt_deferred->getColorBuffer(0)->unbind(Texture2D::SLOT_0);
}



PassGBuffer_Cube::PassGBuffer_Cube()
{
    m_sh_gbuffer = GetShaderGBuffer();
    m_model = GetModelData(MODEL_CUBE);
    m_ubo_instance_pos = GetUniformBufferObject(UBO_CUBE_POS);
    m_instance_pos.reserve(65536);
}

void PassGBuffer_Cube::beforeDraw()
{
    m_instance_pos.clear();
}

void PassGBuffer_Cube::draw()
{
    const uint32 instances_par_batch = 1024;
    const uint32 num_instances = m_instance_pos.size();
    const uint32 num_batch = num_instances/instances_par_batch + (num_instances%instances_par_batch!=0 ? 1 : 0);

    m_ubo_instance_pos->allocate(sizeof(XMVECTOR)*num_instances, UniformBufferObject::USAGE_STREAM, &m_instance_pos[0]);

    for(uint32 i=0; i<num_batch; ++i) {
        uint32 n = std::min<uint32>(num_instances-(i*instances_par_batch), instances_par_batch);
        m_ubo_instance_pos->bindRange(i, sizeof(XMVECTOR)*(instances_par_batch*i), sizeof(XMVECTOR)*n);
        m_sh_gbuffer->setInstancePositionBinding(i);
        m_sh_gbuffer->bind();
        m_model->drawInstanced(n);
        m_sh_gbuffer->unbind();
    }
}


PassDeferred_SphereLight::PassDeferred_SphereLight()
{
    m_sh_deferred = GetShaderDeferred();
    m_model = GetModelData(MODEL_SPHERE);
    m_ubo_instance_pos = GetUniformBufferObject(UBO_LIGHT_POS);
    m_instance_pos.reserve(1024);
}

void PassDeferred_SphereLight::beforeDraw()
{
    m_instance_pos.clear();
}

void PassDeferred_SphereLight::draw()
{
    const uint32 instances_par_batch = 1024;
    const uint32 num_lights = m_instance_pos.size();
    const uint32 num_batch = num_lights/instances_par_batch + (num_lights%instances_par_batch!=0 ? 1 : 0);

    m_ubo_instance_pos->allocate(sizeof(XMVECTOR)*num_lights, UniformBufferObject::USAGE_STREAM, &m_instance_pos[0]);

    for(uint32 i=0; i<num_batch; ++i) {
        uint32 n = std::min<uint32>(num_lights-(i*instances_par_batch), instances_par_batch);
        m_ubo_instance_pos->bindRange(i, sizeof(XMVECTOR)*(instances_par_batch*i), sizeof(XMVECTOR)*n);
        m_sh_deferred->setLightPositionBinding(i);
        m_model->drawInstanced(n);
    }

}

} // namespace atomic
