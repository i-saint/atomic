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
    m_renderer_bloom = AT_NEW(PassPostprocess_Bloom) PassPostprocess_Bloom();
    m_renderers[PASS_GBUFFER].push_back(m_renderer_cube);
    m_renderers[PASS_DEFERRED].push_back(m_renderer_sphere_light);
    m_renderers[PASS_POSTPROCESS].push_back(m_renderer_bloom);

    m_default_viewport.setViewport(0, 0, GetWindowWidth(), GetWindowHeight());
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
    glEnable(GL_DEPTH_TEST);

    uint32 num_renderers = m_renderers[PASS_SHADOW_DEPTH].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_SHADOW_DEPTH][i]->draw();
    }

    glDisable(GL_DEPTH_TEST);
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
    glEnable(GL_DEPTH_TEST);

    uint32 num_renderers = m_renderers[PASS_GBUFFER].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_GBUFFER][i]->draw();
    }

    glDisable(GL_DEPTH_TEST);
    m_sh_gbuffer->unbind();
    m_rt_gbuffer->unbind();
}

void AtomicRenderer::pass_Deferred()
{
    const PerspectiveCamera *camera = GetCamera();
    float aspect_ratio = camera->getAspect();
    vec2 tex_scale = vec2(
        float32(GetWindowWidth())/float32(m_rt_deferred->getWidth()),
        float32(GetWindowHeight())/float32(m_rt_deferred->getHeight()) * aspect_ratio);

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
    m_sh_deferred->setTexcoordScale(tex_scale);
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDepthMask(GL_FALSE);

    uint32 num_renderers = m_renderers[PASS_DEFERRED].size();
    for(uint32 i=0; i<num_renderers; ++i) {
        m_renderers[PASS_DEFERRED][i]->draw();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_DEPTH_TEST);
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
    OrthographicCamera cam;
    cam.setScreen(0.0f, 1.0f, 0.0f, 1.0f);
    cam.bind();

    m_rt_deferred->getColorBuffer(0)->bind(Texture2D::SLOT_0);
    m_sh_out->bind();
    m_sh_out->setColorBuffer(Texture2D::SLOT_0);
    DrawScreen(vec2(0.0f, 0.0f), vec2(float(GetWindowWidth())/float32(m_rt_deferred->getWidth()), float(GetWindowHeight())/float32(m_rt_deferred->getHeight())));
    m_sh_out->unbind();
    m_rt_deferred->getColorBuffer(0)->unbind(Texture2D::SLOT_0);
}



PassGBuffer_Cube::PassGBuffer_Cube()
{
    m_sh_gbuffer = GetShaderGBuffer();
    m_model = GetModelData(MODEL_CUBE);
    m_vbo_instance_pos = GetVertexBufferObject(VBO_CUBE_POS);
    m_instance_pos.reserve(65536);
}

void PassGBuffer_Cube::beforeDraw()
{
    m_instance_pos.clear();
}

void PassGBuffer_Cube::draw()
{
    const uint32 num_instances = m_instance_pos.size();
    m_vbo_instance_pos->allocate(sizeof(XMVECTOR)*num_instances, VertexBufferObject::USAGE_STREAM, &m_instance_pos[0]);

    m_sh_gbuffer->bind();
    m_model->setInstanceData(2, 4, *m_vbo_instance_pos);
    m_model->drawInstanced(num_instances);
    m_sh_gbuffer->unbind();
}


PassDeferred_SphereLight::PassDeferred_SphereLight()
{
    m_sh_deferred = GetShaderDeferred();
    m_model = GetModelData(MODEL_SPHERE);
    m_vbo_instance_pos = GetVertexBufferObject(VBO_SPHERE_LIGHT_POS);
    m_instance_pos.reserve(1024);
}

void PassDeferred_SphereLight::beforeDraw()
{
    m_instance_pos.clear();
}

void PassDeferred_SphereLight::draw()
{
    const uint32 num_instances = m_instance_pos.size();
    m_vbo_instance_pos->allocate(sizeof(XMVECTOR)*num_instances, VertexBufferObject::USAGE_STREAM, &m_instance_pos[0]);

    m_model->setInstanceData(2, 4, *m_vbo_instance_pos);
    m_model->drawInstanced(num_instances);

}


PassPostprocess_Bloom::PassPostprocess_Bloom()
: m_rt_deferred(NULL)
, m_rt_gauss0(NULL)
, m_rt_gauss1(NULL)
, m_sh_bloom(NULL)
{
    m_rt_deferred = GetRenderTargetDeferred();
    m_rt_gauss0 = GetRenderTargetGauss(0);
    m_rt_gauss1 = GetRenderTargetGauss(1);
    m_sh_bloom = GetShaderBloom();
}

void PassPostprocess_Bloom::beforeDraw()
{
}

void PassPostprocess_Bloom::draw()
{
    const float32 aspect_ratio = GetWindowAspectRatio();
    Viewport viewports[] = {
        Viewport(  0,0, 256,256),
        Viewport(256,0, 128,128),
        Viewport(384,0,  64, 64),
        Viewport(448,0,  32, 32),
    };
    vec2 texcoord_pos[] = {
        vec2( 0.0f, 0.0f),
        vec2( 0.5f, 0.0f),
        vec2(0.75f, 0.0f),
        vec2(0.875f, 0.0f),
    };
    vec2 texcoord_size[] = {
        vec2(  0.5f,  1.0f),
        vec2( 0.25f,  0.5f),
        vec2(0.125f, 0.25f),
        vec2(0.0625f, 0.125f),
    };

    OrthographicCamera cam;
    cam.setScreen(0.0f, 1.0f, 0.0f, 1.0f);
    cam.bind();

    m_sh_bloom->bind();
    m_sh_bloom->setScreenWidth((float32)m_rt_gauss0->getWidth());
    m_sh_bloom->setScreenHeight((float32)m_rt_gauss0->getHeight());
    // ‹P“x’Šo
    {
        m_sh_bloom->switchToPickupPass();
        m_rt_gauss0->bind();
        m_rt_deferred->getColorBuffer(0)->bind(Texture2D::SLOT_0);
        m_sh_bloom->setColorBuffer(Texture2D::SLOT_0);
        for(uint32 i=0; i<_countof(viewports); ++i) {
            viewports[i].bind();
            DrawScreen();
        }
        // todo
        m_rt_deferred->getColorBuffer(0)->unbind();
        m_rt_gauss0->unbind();
    }

    // ‰¡ƒuƒ‰[
    {
        m_sh_bloom->switchToHorizontalBlurPass();
        m_rt_gauss1->bind();
        m_rt_gauss0->getColorBuffer(0)->bind(Texture2D::SLOT_0);
        m_sh_bloom->setColorBuffer(Texture2D::SLOT_0);
        for(uint32 i=0; i<_countof(viewports); ++i) {
            viewports[i].bind();
            vec2 tmin = texcoord_pos[i];
            vec2 tmax = texcoord_pos[i]+texcoord_size[i];
            tmax.y /= aspect_ratio;
            m_sh_bloom->setTexcoordMin(tmin);
            m_sh_bloom->setTexcoordMax(tmax);
            DrawScreen(texcoord_pos[i], texcoord_pos[i]+texcoord_size[i]);
        }
        m_rt_gauss0->getColorBuffer(0)->unbind();
        m_rt_gauss1->unbind();
    }

    // cƒuƒ‰[
    {
        m_sh_bloom->switchToVerticalBlurPass();
        m_rt_gauss0->bind();
        m_rt_gauss1->getColorBuffer(0)->bind(Texture2D::SLOT_0);
        m_sh_bloom->setColorBuffer(Texture2D::SLOT_0);
        for(uint32 i=0; i<_countof(viewports); ++i) {
            viewports[i].bind();
            vec2 tmin = texcoord_pos[i];
            vec2 tmax = texcoord_pos[i]+texcoord_size[i];
            tmax.y /= aspect_ratio;
            m_sh_bloom->setTexcoordMin(tmin);
            m_sh_bloom->setTexcoordMax(tmax);
            DrawScreen(texcoord_pos[i], texcoord_pos[i]+texcoord_size[i]);
        }
        m_rt_gauss1->getColorBuffer(0)->unbind();
        m_rt_gauss0->unbind();
    }

    // ‰ÁŽZ
    GetDefaultViewport()->bind();
    {
        m_sh_bloom->switchToCompositePass();
        m_rt_deferred->bind();
        m_rt_gauss0->getColorBuffer(0)->bind(Texture2D::SLOT_0);
        m_sh_bloom->setColorBuffer(Texture2D::SLOT_0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        for(uint32 i=0; i<_countof(viewports); ++i) {
            vec2 tmin = texcoord_pos[i];
            vec2 tmax = texcoord_pos[i]+texcoord_size[i];
            tmax.y /= aspect_ratio;
            DrawScreen(tmin, tmax);
        }
        glDisable(GL_BLEND);
        m_rt_gauss0->getColorBuffer(0)->unbind();
        m_rt_deferred->unbind();
    }
    m_sh_bloom->unbind();

    GetDefaultViewport()->bind();

}

} // namespace atomic
