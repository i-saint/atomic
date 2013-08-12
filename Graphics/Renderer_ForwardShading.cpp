#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/CollisionModule.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atm {



PassForward_Generic::PassForward_Generic()
    : m_frame()
{
}

PassForward_Generic::~PassForward_Generic()
{
}

void PassForward_Generic::beforeDraw()
{
    for(auto si=m_commands.begin(); si!=m_commands.end(); ++si) {
        ModelParamCont &mm = si->second;
        for(auto mi=mm.begin(); mi!=mm.end(); ++mi) {
            mi->second.clear();
        }
    }
}

void PassForward_Generic::draw()
{
    if(m_commands.empty()) { return; }
    ++m_frame;

    static const VBO_RID s_vboids[] = {VBO_GENERIC_PARAMS1, VBO_GENERIC_PARAMS2};
    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    RenderTarget *rt = atmGetBackRenderTarget();
    Buffer *vbo_params = atmGetVertexBuffer(s_vboids[m_frame%2]);
    rt->setDepthStencilBuffer(atmGetRenderTarget(RT_GBUFFER)->getDepthStencilBuffer());
    dc->setBlendState(atmGetBlendState(BS_BLEND_ALPHA));
    dc->setDepthStencilState(atmGetDepthStencilState(DS_DEPTH_ENABLED));
    dc->setRenderTarget(rt);

    for(auto si=m_commands.begin(); si!=m_commands.end(); ++si) {
        const ModelParamCont &mm = si->second;
        for(auto mi=mm.begin(); mi!=mm.end(); ++mi) {
            m_params.insert(m_params.end(), mi->second.begin(), mi->second.end());
        }
    }
    if(!m_params.empty()) {
        size_t capacity = vbo_params->getDesc().size;
        size_t size_byte = sizeof(InstanceParams)*m_params.size();
        istAssert(size_byte < capacity);
        MapAndWrite(dc, vbo_params, &m_params[0], std::min<size_t>(size_byte, capacity));
        m_params.clear();

        static const VertexDesc transform_descs[] = {
            {GLSL_INSTANCE_TRANSFORM1, I3D_FLOAT32,4,  0, false, 1},
            {GLSL_INSTANCE_TRANSFORM2, I3D_FLOAT32,4, 16, false, 1},
            {GLSL_INSTANCE_TRANSFORM3, I3D_FLOAT32,4, 32, false, 1},
            {GLSL_INSTANCE_TRANSFORM4, I3D_FLOAT32,4, 48, false, 1},
            {GLSL_INSTANCE_PARAM1,     I3D_FLOAT32,4, 64, false, 1},
            {GLSL_INSTANCE_PARAM2,     I3D_FLOAT32,4, 80, false, 1},
            {GLSL_INSTANCE_PARAM3,     I3D_FLOAT32,4, 96, false, 1},
            {GLSL_INSTANCE_PARAM4,     I3D_FLOAT32,4,112, false, 1},
        };
        size_t params_offset = 0;
        for(auto si=m_commands.begin(); si!=m_commands.end(); ++si) {
            const ModelParamCont &mm = si->second;

            AtomicShader *sh = atmGetShader(si->first);
            sh->bind();
            for(auto mi=mm.begin(); mi!=mm.end(); ++mi) {
                if(mi->second.empty()) { continue; }

                const ModelInfo &model = *atmGetModelInfo(mi->first);
                const ParamCont &params = mi->second;
                VertexArray *va = atmGetVertexArray(model.vertices);
                Buffer *ibo = atmGetIndexBuffer(model.indices);
                dc->setIndexBuffer(ibo, 0, I3D_UINT32);
                va->setAttributes(1, vbo_params, sizeof(InstanceParams)*params_offset, sizeof(InstanceParams), transform_descs, _countof(transform_descs));
                dc->setVertexArray(va);
                if(ibo) {
                    dc->drawIndexedInstanced(model.topology, 0, model.num_indices, params.size());
                }
                else {
                    dc->drawInstanced(model.topology, 0, model.num_indices, params.size());
                }
                dc->setIndexBuffer(nullptr, 0, I3D_UINT32);
                dc->setVertexArray(nullptr);

                params_offset += params.size();
            }
            sh->unbind();
        }
    }

    dc->setDepthStencilState(atmGetDepthStencilState(DS_NO_DEPTH_NO_STENCIL));
    dc->setBlendState(atmGetBlendState(BS_NO_BLEND));
    rt->setDepthStencilBuffer(nullptr);
}

void PassForward_Generic::drawModel( SH_RID shader, MODEL_RID model, const mat4 &trans )
{
    InstanceParams params;
    params.transform = trans;
    m_commands[shader][model].push_back(params);
}

void PassForward_Generic::drawModel( SH_RID shader, MODEL_RID model, const InstanceParams &params )
{
    m_commands[shader][model].push_back(params);
}



PassForward_Indicators::PassForward_Indicators()
{
}

PassForward_Indicators::~PassForward_Indicators()
{
}

void PassForward_Indicators::beforeDraw()
{
}

void PassForward_Indicators::draw()
{
	//return;
    if(!atmGetGame()) { return; }

    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    VertexArray  *va_grid = atmGetVertexArray(VA_FIELD_GRID);
    AtomicShader *sh_grid = atmGetShader(SH_FILL3D);
    RenderTarget *rt = atmGetBackRenderTarget();

    rt->setDepthStencilBuffer(atmGetRenderTarget(RT_GBUFFER)->getDepthStencilBuffer());
    dc->setBlendState(atmGetBlendState(BS_BLEND_ALPHA));
    dc->setDepthStencilState(atmGetDepthStencilState(DS_DEPTH_ENABLED));
    dc->setRenderTarget(rt);

    sh_grid->bind();
    dc->setVertexArray(va_grid);
    dc->draw(I3D_LINES, 0, 6*6*2);
    dc->setVertexArray(nullptr);
    sh_grid->unbind();

    dc->setDepthStencilState(atmGetDepthStencilState(DS_NO_DEPTH_NO_STENCIL));
    dc->setBlendState(atmGetBlendState(BS_NO_BLEND));
    rt->setDepthStencilBuffer(nullptr);
}



PassForward_Barrier::PassForward_Barrier()
{
    m_solids.vbo    = VBO_FW_RIGID_BARRIER;
    m_solids.shader = SH_BARRIER_PARTICLES;
    m_solids.params = TEX2D_PSET_PARAMS_FW_BARRIER;
}

PassForward_Barrier::~PassForward_Barrier()
{

}

void PassForward_Barrier::beforeDraw()
{
    m_solids.clear();
}

void PassForward_Barrier::draw()
{
    PassGBuffer_Fluid::drawParticleSets(m_solids);
}

void PassForward_Barrier::addParticles( PSET_RID psid, const PSetInstance &inst, uint32 n )
{
    if(!PassGBuffer_Fluid::culling(psid, inst)) { return; }

    const ParticleSet *rc = atmGetParticleSet(psid);
    uint32 num_particles = rc->getNumParticles();
    PSetUpdateInfo tmp;
    tmp.psid        = psid;
    tmp.instanceid  = m_solids.instance_data.size();
    tmp.num = n!=0 ? std::min(n, num_particles) : num_particles;
    m_solids.update_info.push_back(tmp);
    m_solids.instance_data.push_back(inst);
}





PassForward_Background::PassForward_Background()
    : m_shader(SH_BG2)
{
    wdmAddNode("Rendering/BG/Enable", &m_shader, (int32)SH_BG1, (int32)SH_BG_END);
}

PassForward_Background::~PassForward_Background()
{
    wdmEraseNode("Rendering/BG");
}

void PassForward_Background::beforeDraw()
{
}

void PassForward_Background::draw()
{
    if(atmGetConfig()->bg_level==atmE_BGNone) { return; }

    i3d::DeviceContext *dc  = atmGetGLDeviceContext();
    AtomicShader *sh_bg     = atmGetShader((SH_RID)m_shader);
    AtomicShader *sh_up     = atmGetShader(SH_GBUFFER_UPSAMPLING);
    AtomicShader *sh_out    = atmGetShader(SH_OUTPUT);
    VertexArray *va_quad    = atmGetVertexArray(VA_SCREEN_QUAD);
    RenderTarget *brt       = atmGetBackRenderTarget();
    RenderTarget *frt       = atmGetFrontRenderTarget();
    RenderTarget *bgrt      = atmGetRenderTarget(RT_OUTPUT2);

    Buffer *ubo_rs          = atmGetUniformBuffer(UBO_RENDERSTATES_3D);
    RenderStates *rs        = atmGetRenderStates();

    if(atmGetConfig()->bg_multiresolution) {
        // 1/4 の解像度で raymarching
        rs->ScreenSize      = vec2(atmGetWindowSize())/4.0f;
        rs->RcpScreenSize   = vec2(1.0f, 1.0f) / rs->ScreenSize;
        MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));

        dc->setViewport(Viewport(ivec2(), brt->getColorBuffer(0)->getDesc().size/4U));
        dc->setRenderTarget(nullptr);
        dc->generateMips(brt->getDepthStencilBuffer());
        brt->setMipmapLevel(2);
        //dc->clearDepthStencil(gbuffer, 1.0f, 0);
        dc->setRenderTarget(brt);

        sh_bg->bind();
        dc->setVertexArray(va_quad);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_BG));
        dc->draw(I3D_QUADS, 0, 4);
        sh_bg->unbind();

        dc->setRenderTarget(nullptr);
        brt->setMipmapLevel(0);
        dc->setRenderTarget(brt);
        dc->setViewport(Viewport(ivec2(), brt->getColorBuffer(0)->getDesc().size));

        rs->ScreenSize      = vec2(atmGetWindowSize());
        rs->RcpScreenSize   = vec2(1.0f, 1.0f) / rs->ScreenSize;
        MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));


        // 変化量少ない部分を upsampling
        dc->setTexture(GLSL_COLOR_BUFFER, brt->getColorBuffer(GBUFFER_COLOR));
        dc->setTexture(GLSL_NORMAL_BUFFER, brt->getColorBuffer(GBUFFER_NORMAL));
        dc->setTexture(GLSL_POSITION_BUFFER, brt->getColorBuffer(GBUFFER_POSITION));
        dc->setTexture(GLSL_GLOW_BUFFER, brt->getColorBuffer(GBUFFER_GLOW));
        dc->setVertexArray(va_quad);
        sh_up->bind();
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_UPSAMPLING));
        dc->draw(I3D_QUADS, 0, 4);
        sh_up->unbind();
        dc->setTexture(GLSL_COLOR_BUFFER, nullptr);
        dc->setTexture(GLSL_NORMAL_BUFFER, nullptr);
        dc->setTexture(GLSL_POSITION_BUFFER, nullptr);
        dc->setTexture(GLSL_GLOW_BUFFER, nullptr);
    }

    {
        int resx = 1;
        switch(atmGetConfig()->bg_level) {
        case atmE_BGResolution_x1: resx=1; break;
        case atmE_BGResolution_x2: resx=2; break;
        case atmE_BGResolution_x4: resx=4; break;
        case atmE_BGResolution_x8: resx=8; break;
        }
        if(resx!=1) {
            rs->ScreenSize      = vec2(atmGetWindowSize())/(float32)resx;
            rs->RcpScreenSize   = vec2(1.0f, 1.0f) / rs->ScreenSize;
            MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));

            dc->setViewport(Viewport(ivec2(), brt->getColorBuffer(0)->getDesc().size/(uint32)resx));
            //bgrt->setDepthStencilBuffer(brt->getDepthStencilBuffer()); // 縮小しないといけない
            dc->setRenderTarget(bgrt);
        }

        sh_bg->bind();
        dc->setVertexArray(va_quad);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_BG));
        dc->draw(I3D_QUADS, 0, 4);
        dc->setDepthStencilState(atmGetDepthStencilState(DS_NO_DEPTH_NO_STENCIL));
        sh_bg->unbind();

        if(resx!=1) {
            rs->ScreenSize      = vec2(atmGetWindowSize());
            rs->RcpScreenSize   = vec2(1.0f, 1.0f) / rs->ScreenSize;
            rs->ScreenTexcoord  = vec2(1.0f, 1.0f) / float32(resx);
            MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));
            dc->setViewport(Viewport(ivec2(), brt->getColorBuffer(0)->getDesc().size));

            sh_out->bind();
            dc->setTexture(GLSL_COLOR_BUFFER, bgrt->getColorBuffer(0));
            dc->setRenderTarget(brt);
            dc->setDepthStencilState(atmGetDepthStencilState(DS_GBUFFER_BG));
            dc->draw(I3D_QUADS, 0, 4);
            dc->setDepthStencilState(atmGetDepthStencilState(DS_NO_DEPTH_NO_STENCIL));
            dc->setTexture(GLSL_COLOR_BUFFER, nullptr);
            sh_out->unbind();

            rs->ScreenTexcoord  = rs->ScreenSize / vec2(brt->getColorBuffer(0)->getDesc().size);
            MapAndWrite(dc, ubo_rs, rs, sizeof(*rs));
            bgrt->setDepthStencilBuffer(nullptr);
        }
    }

    {
        sh_out->assign(dc);
        dc->setRenderTarget(atmGetPrevBackbuffer());
        dc->setTexture(GLSL_COLOR_BUFFER, brt->getColorBuffer(0));
        dc->draw(I3D_QUADS, 0, 4);
        dc->setRenderTarget(brt);
    }
}


} // namespace atm
