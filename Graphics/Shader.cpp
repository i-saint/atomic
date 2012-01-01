#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Shader.h"
#include "Renderer.h"
#include "AtomicRenderingSystem.h"
#include "shader/glsl_source.h"

namespace atomic {

AtomicShader::AtomicShader()
: m_shader(NULL)
, m_vs(NULL)
, m_ps(NULL)
, m_loc_renderstates(0)
{
    i3d::Device *dev = atomicGetGraphicsDevice();
    m_shader = dev->createShaderProgram();
    m_vs = dev->createVertexShader();
    m_ps = dev->createPixelShader();
}

AtomicShader::~AtomicShader()
{
    atomicSafeRelease(m_shader);
    atomicSafeRelease(m_vs);
    atomicSafeRelease(m_ps);
}

void AtomicShader::release()
{
    istDelete(this);
}

bool AtomicShader::loadFromMemory( const char* src )
{
    CreateVertexShaderFromString(*m_vs, src);
    CreateFragmentShaderFromString(*m_ps, src);
    m_shader->link(m_vs, m_ps, NULL);

    m_loc_renderstates = m_shader->getUniformBlockIndex("render_states");

#define SetSampler(name, value) { GLint l=m_shader->getUniformLocation(name); if(l!=-1){ m_shader->setUniform1i(l, value); }}
    m_shader->bind();
    SetSampler("u_ColorBuffer",     GLSL_COLOR_BUFFER);
    SetSampler("u_NormalBuffer",    GLSL_NORMAL_BUFFER);
    SetSampler("u_PositionBuffer",  GLSL_POSITION_BUFFER);
    SetSampler("u_GlowBuffer",      GLSL_GLOW_BUFFER);
    SetSampler("u_RandomBuffer",    GLSL_RANDOM_BUFFER);
    m_shader->unbind();
#undef SetSampler

    return true;
}

GLint AtomicShader::getUniformBlockIndex(const char *name) const
{
    return m_shader->getUniformBlockIndex(name);
}

void AtomicShader::setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, GLuint uniformBufferHandle)
{
    m_shader->setUniformBlock(uniformBlockIndex, uniformBindingIndex, uniformBufferHandle);
}

void AtomicShader::bind()
{
    m_shader->bind();
    m_shader->setUniformBlock(m_loc_renderstates, GLSL_RENDERSTATE_BINDING, atomicGetUniformBufferObject(UBO_RENDER_STATES)->getHandle());
}

void AtomicShader::unbind()
{
    m_shader->unbind();
}



} // namespace atomic