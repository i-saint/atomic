#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Shader.h"
#include "Renderer.h"
#include "shader/glsl_source.h"

namespace atomic {

AtomicShader::AtomicShader()
: m_loc_renderstates(0)
, m_rs_binding(0)
{
    static int32 s_gen = 0;
    m_rs_binding = s_gen++;
}

bool AtomicShader::initialize()
{
    //m_program.initialize();
    return super::initialize() && m_vsh.initialize() && m_fsh.initialize();
}

void AtomicShader::finalize()
{
    m_vsh.finalize();
    m_fsh.finalize();
    super::finalize();
}

bool AtomicShader::loadFromMemory( const char* src )
{
    CreateVertexShaderFromString(m_vsh, src);
    CreateFragmentShaderFromString(m_fsh, src);
    link(&m_vsh, &m_fsh, NULL);

    m_loc_renderstates = getUniformBlockIndex("render_states");
    return true;
}

void AtomicShader::bind()
{
    super::bind();
    setUniformBlock(m_loc_renderstates, GLSL_RENDERSTATE_BINDING, atomicGetUniformBufferObject(UBO_RENDER_STATES)->getHandle());
}



bool ShaderDeferred::loadFromMemory( const char* src )
{
    super::loadFromMemory(src);

    m_loc_color_buffer      = getUniformLocation("u_ColorBuffer");
    //m_loc_glow_buffer       = getUniformLocation("u_GlowBuffer"); // ‚ ‚Æ‚Å
    m_loc_normal_buffer     = getUniformLocation("u_NormalBuffer");
    m_loc_position_buffer   = getUniformLocation("u_PositionBuffer");
    //m_loc_depth_buffer      = getUniformLocation("u_DepthBuffer");

    m_loc_rcp_aspect_ratio  = getUniformLocation("u_RcpAspectRatio");
    m_loc_texcoord_scale    = getUniformLocation("u_TexcoordScale");

    return true;
}


bool ShaderBloom::initialize()
{
    super::initialize();
    m_vsh.initialize();
    m_fsh.initialize();
    CreateVertexShaderFromString(m_vsh, g_bloom_vsh);
    CreateFragmentShaderFromString(m_fsh, g_bloom_fsh);
    link(&m_vsh, &m_fsh, NULL);

    m_loc_color_buffer      = getUniformLocation("u_ColorBuffer");
    m_loc_rcp_screen_width  = getUniformLocation("u_RcpScreenWidth");
    m_loc_rcp_screen_height = getUniformLocation("u_RcpScreenHeight");
    m_loc_texcoord_min      = getUniformLocation("u_TexcoordMin");
    m_loc_texcoord_max      = getUniformLocation("u_TexcoordMax");

    m_sub_pickup            = getSubroutineIndexF("pickup");
    m_sub_hblur             = getSubroutineIndexF("horizontalBlur");
    m_sub_vblur             = getSubroutineIndexF("verticalBlur");
    m_sub_composite         = getSubroutineIndexF("composite");

    return true;
}



} // namespace atomic