#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/Collision.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atm {


PassForwardShading_DistanceField::PassForwardShading_DistanceField()
{
    m_sh_grid       = atmGetShader(SH_FILL);
    m_va_grid       = atmGetVertexArray(VA_FIELD_GRID);

    m_sh_cell       = atmGetShader(SH_DISTANCE_FIELD);
    m_vbo_cell_pos  = atmGetVertexBuffer(VBO_DISTANCE_FIELD_POS);
    m_vbo_cell_dist = atmGetVertexBuffer(VBO_DISTANCE_FIELD_DIST);
    m_va_cell       = atmGetVertexArray(VA_DISTANCE_FIELD);
}

void PassForwardShading_DistanceField::beforeDraw()
{
}

void PassForwardShading_DistanceField::draw()
{
    i3d::DeviceContext *dc = atmGetGLDeviceContext();
#ifdef atm_enable_distance_field
    if(atmGetConfig()->debug_show_distance) {
        MapAndWrite(*m_vbo_cell_dist, atmGetCollisionSet()->getDistanceField()->getDistances(),
            sizeof(vec4) * SPH_DISTANCE_FIELD_DIV_X * SPH_DISTANCE_FIELD_DIV_Y);
        m_sh_cell->bind();
        m_va_cell->bind();
        glDrawArraysInstanced(GL_QUADS, 0, 4, SPH_DISTANCE_FIELD_DIV_X * SPH_DISTANCE_FIELD_DIV_Y);
        m_va_cell->unbind();
        m_sh_cell->unbind();
    }
#endif // atm_enable_distance_field

    if(atmGetConfig()->debug_show_grid) {
        m_sh_grid->bind();
        dc->setVertexArray(m_va_grid);
        dc->draw(I3D_LINES, 0, (PSYM_GRID_DIV+1) * (PSYM_GRID_DIV+1) * 2);
        dc->setVertexArray(NULL);
        m_sh_grid->unbind();
    }
}




Pass_BackGround::Pass_BackGround()
    : m_shader(SH_BG1)
{
    wdmAddNode("Rendering/BG/Enable", &m_shader, (int32)SH_BG1, (int32)SH_BG_END);
}

Pass_BackGround::~Pass_BackGround()
{
    wdmEraseNode("Rendering/BG");
}

void Pass_BackGround::beforeDraw()
{
}

void Pass_BackGround::draw()
{
    if(!atmGetConfig()->bg) { return; }

    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    AtomicShader *sh_bg     = atmGetShader((SH_RID)m_shader);
    AtomicShader *sh_up     = atmGetShader(SH_GBUFFER_UPSAMPLING);
    VertexArray *va_quad    = atmGetVertexArray(VA_SCREEN_QUAD);
    RenderTarget *gbuffer   = atmGetFrontRenderTarget();

    Buffer *ubo_rs          = atmGetUniformBuffer(UBO_RENDERSTATES_3D);
    RenderStates *rs        = atmGetRenderStates();


    if(atmGetConfig()->bg_multiresolution) {
        // 1/4 の解像度で raymarching
        rs->ScreenSize      = vec2(atmGetWindowSize())/4.0f;
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
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_BG));
        dc->draw(I3D_QUADS, 0, 4);
        sh_bg->unbind();

        dc->setRenderTarget(NULL);
        gbuffer->setMipmapLevel(0);
        dc->setRenderTarget(gbuffer);
        dc->setViewport(Viewport(ivec2(), gbuffer->getColorBuffer(0)->getDesc().size));

        rs->ScreenSize      = vec2(atmGetWindowSize());
        rs->RcpScreenSize   = vec2(1.0f, 1.0f) / rs->ScreenSize;
        MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));


        // 変化量少ない部分を upsampling
        dc->setTexture(GLSL_COLOR_BUFFER, gbuffer->getColorBuffer(GBUFFER_COLOR));
        dc->setTexture(GLSL_NORMAL_BUFFER, gbuffer->getColorBuffer(GBUFFER_NORMAL));
        dc->setTexture(GLSL_POSITION_BUFFER, gbuffer->getColorBuffer(GBUFFER_POSITION));
        dc->setTexture(GLSL_GLOW_BUFFER, gbuffer->getColorBuffer(GBUFFER_GLOW));
        dc->setVertexArray(va_quad);
        sh_up->bind();
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_UPSAMPLING));
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
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_BG));
        dc->draw(I3D_QUADS, 0, 4);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_NO_DEPTH_NO_STENCIL));
        sh_bg->unbind();
    }
}

} // namespace atm
