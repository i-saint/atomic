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

#define SetSampler(name, value) { GLint l=getUniformLocation(name); if(l!=-1){ setUniform1i(l, value); }}
    super::bind();
    SetSampler("u_ColorBuffer",     GLSL_COLOR_BUFFER);
    SetSampler("u_NormalBuffer",    GLSL_NORMAL_BUFFER);
    SetSampler("u_PositionBuffer",  GLSL_POSITION_BUFFER);
    SetSampler("u_DepthBuffer",     GLSL_DEPTH_BUFFER);
    SetSampler("u_RandomBuffer",    GLSL_RANDOM_BUFFER);
    super::unbind();
#undef SetSampler

    return true;
}

void AtomicShader::bind()
{
    super::bind();
    setUniformBlock(m_loc_renderstates, GLSL_RENDERSTATE_BINDING, atomicGetUniformBufferObject(UBO_RENDER_STATES)->getHandle());
}



} // namespace atomic