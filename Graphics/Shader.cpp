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

    m_block_instance_position = getUniformBlockIndex("InstancePosition");

    return true;
}




bool ShaderDeferred::initialize()
{
    CreateVertexShaderFromFile(m_vsh, "shader/deferred.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/deferred.fsh");
    super::initialize(&m_vsh, NULL, &m_fsh);

    m_loc_color_buffer      = getUniformLocation("ColorBuffer");
    //m_loc_glow_buffer       = getUniformLocation("GlowBuffer"); // ‚ ‚Æ‚Å
    m_loc_normal_buffer     = getUniformLocation("NormalBuffer");
    m_loc_position_buffer   = getUniformLocation("PositionBuffer");
    //m_loc_depth_buffer      = getUniformLocation("DepthBuffer");

    m_loc_aspect_ratio      = getUniformLocation("AspectRatio");
    m_loc_texcoord_scale    = getUniformLocation("TexcoordScale");

    m_block_light_position = getUniformBlockIndex("LightPosition");

    return true;
}

bool ShaderOutput::initialize()
{
    CreateVertexShaderFromFile(m_vsh, "shader/out.vsh");
    CreateFragmentShaderFromFile(m_fsh, "shader/out.fsh");
    super::initialize(&m_vsh, NULL, &m_fsh);

    return true;
}


} // namespace atomic