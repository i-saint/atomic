#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Shader.h"

namespace atomic {


bool ShaderGBuffer::initialize()
{
    CreateVertexShaderFromFile(m_vsh, "shader/gbuffer.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/gbuffer.fsh");
    super::initialize(&m_vsh, NULL, &m_fsh);

    return true;
}


bool ShaderGBuffer_Octahedron::initialize()
{
    CreateVertexShaderFromFile(m_vsh, "shader/gbuffer_octahedron.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/gbuffer_octahedron.fsh");
    super::initialize(&m_vsh, NULL, &m_fsh);

    return true;
}



bool ShaderDeferred::initialize()
{
    CreateVertexShaderFromFile(m_vsh, "shader/deferred.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/deferred.fsh");
    super::initialize(&m_vsh, NULL, &m_fsh);

    m_loc_color_buffer      = getUniformLocation("u_ColorBuffer");
    //m_loc_glow_buffer       = getUniformLocation("u_GlowBuffer"); // ‚ ‚Æ‚Å
    m_loc_normal_buffer     = getUniformLocation("u_NormalBuffer");
    m_loc_position_buffer   = getUniformLocation("u_PositionBuffer");
    //m_loc_depth_buffer      = getUniformLocation("u_DepthBuffer");

    m_loc_aspect_ratio      = getUniformLocation("u_AspectRatio");
    m_loc_texcoord_scale    = getUniformLocation("u_TexcoordScale");

    return true;
}


bool ShaderBloom::initialize()
{
    CreateVertexShaderFromFile(m_vsh, "shader/bloom.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/bloom.fsh");
    super::initialize(&m_vsh, NULL, &m_fsh);

    m_loc_color_buffer      = getUniformLocation("u_ColorBuffer");
    m_loc_screen_width      = getUniformLocation("u_ScreenWidth");
    m_loc_screen_height     = getUniformLocation("u_ScreenHeight");
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
    CreateVertexShaderFromFile(m_vsh, "shader/out.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/out.fsh");
    super::initialize(&m_vsh, NULL, &m_fsh);

    m_loc_color_buffer      = getUniformLocation("u_ColorBuffer");

    return true;
}


} // namespace atomic