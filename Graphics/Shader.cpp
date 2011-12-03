#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Shader.h"
#include "shader/glsl_source.h"

namespace atomic {


bool ShaderGBuffer::initialize()
{
    CreateVertexShaderFromString(m_vsh, g_gbuffer_glsl);
    CreateFragmentShaderFromString(m_fsh, g_gbuffer_glsl);
    super::initialize(&m_vsh, NULL, &m_fsh);

    return true;
}


bool ShaderGBuffer_Octahedron::initialize()
{
    CreateVertexShaderFromString(m_vsh, g_gbuffer_octahedron_vsh);
    CreateFragmentShaderFromString(m_fsh, g_gbuffer_octahedron_fsh);
    super::initialize(&m_vsh, NULL, &m_fsh);

    return true;
}



bool ShaderDeferred::initialize()
{
    CreateVertexShaderFromString(m_vsh, g_deferred_vsh);
    CreateFragmentShaderFromString(m_fsh, g_deferred_fsh);
    super::initialize(&m_vsh, NULL, &m_fsh);

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
    CreateVertexShaderFromString(m_vsh, g_bloom_vsh);
    CreateFragmentShaderFromString(m_fsh, g_bloom_fsh);
    super::initialize(&m_vsh, NULL, &m_fsh);

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


bool ShaderOutput::initialize()
{
    CreateVertexShaderFromString(m_vsh, g_out_vsh);
    CreateFragmentShaderFromString(m_fsh, g_out_fsh);
    super::initialize(&m_vsh, NULL, &m_fsh);

    m_loc_color_buffer      = getUniformLocation("u_ColorBuffer");

    return true;
}


} // namespace atomic