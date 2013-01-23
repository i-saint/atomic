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
, m_gs(NULL)
, m_loc_renderstates(0)
{
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
    i3d::Device *dev = atomicGetGLDevice();
    {
        stl::string source;
        source += "#version 330 core\n";
        source += "#define GLSL\n";
        source += "#define GLSL_VS\n";
        source += src;
        VertexShaderDesc desc = VertexShaderDesc(source.c_str(), source.size());
        m_vs = dev->createVertexShader(desc);
    }
    {
        stl::string source;
        source += "#version 330 core\n";
        source += "#define GLSL\n";
        source += "#define GLSL_PS\n";
        source += src;
        PixelShaderDesc desc = PixelShaderDesc(source.c_str(), source.size());
        m_ps = dev->createPixelShader(desc);
    }
    //{
    //    stl::string source;
    //    source += "#define GLSL\n";
    //    source += "#define GLSL_GS\n";
    //    source += src;
    //    GeometryShaderDesc desc = GeometryShaderDesc(source.c_str(), source.size());
    //    m_gs = dev->createGeometryShader(desc);
    //}
    {
        ShaderProgramDesc desc(m_vs, m_ps, m_gs);
        m_shader = dev->createShaderProgram(desc);
    }


    m_loc_renderstates = m_shader->getUniformBlockIndex("render_states");

    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
#define SetSampler(name, value) { GLint l=m_shader->getUniformLocation(name); if(l!=-1){ m_shader->setSampler(l, value); }}
    dc->setShader(m_shader);
    SetSampler("u_ColorBuffer",     GLSL_COLOR_BUFFER);
    SetSampler("u_NormalBuffer",    GLSL_NORMAL_BUFFER);
    SetSampler("u_PositionBuffer",  GLSL_POSITION_BUFFER);
    SetSampler("u_GlowBuffer",      GLSL_GLOW_BUFFER);
    SetSampler("u_BackBuffer",      GLSL_BACK_BUFFER);
    SetSampler("u_RandomBuffer",    GLSL_RANDOM_BUFFER);
    SetSampler("u_ParamBuffer",     GLSL_PARAM_BUFFER);
    dc->setShader(NULL);
#undef SetSampler

    return true;
}

GLint AtomicShader::getUniformBlockIndex(const char *name) const
{
    return m_shader->getUniformBlockIndex(name);
}

void AtomicShader::setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, Buffer *buffer)
{
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    dc->setUniformBuffer(uniformBlockIndex, uniformBindingIndex, buffer);
}

void AtomicShader::bind()
{
    i3d::DeviceContext *dc = atomicGetGLDeviceContext();
    assign(dc);
}

void AtomicShader::unbind()
{
    i3d::DeviceContext *ctx = atomicGetGLDeviceContext();
    ctx->setShader(NULL);
}

void AtomicShader::assign( i3d::DeviceContext *dc )
{
    dc->setShader(m_shader);
    dc->setUniformBuffer(m_loc_renderstates, GLSL_RENDERSTATE_BINDING, atomicGetUniformBuffer(UBO_RENDERSTATES_3D));
}



} // namespace atomic