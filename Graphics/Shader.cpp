#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Shader.h"
#include "Renderer.h"
#include "AtomicRenderingSystem.h"

namespace atm {


AtomicShader::AtomicShader()
: m_shader(nullptr)
, m_loc_renderstates(0)
#ifdef atm_enable_ShaderLiveEdit
, m_timestamp(0)
#endif // atm_enable_ShaderLiveEdit
{
}

AtomicShader::~AtomicShader()
{
    clearShaders();
}

void AtomicShader::release()
{
    istDelete(this);
}

void AtomicShader::clearShaders()
{
    atmSafeRelease(m_shader);
    m_loc_renderstates = 0;
}

bool AtomicShader::createShaders( const char *filename )
{
    i3d::Device *dev = atmGetGLDevice();
    ShaderProgramDesc sh_desc;

#ifdef atm_enable_ShaderLiveEdit
    m_glsl_filename = filename;
    static const char s_glsl_path[] = "shader/";
    static const char s_shader_path[] = "shader/tmp/";
    {
        Poco::File glsl((stl::string(s_glsl_path)+filename+".glsl").c_str());
        m_timestamp = glsl.getLastModified();
    }
    {
        stl::string vs_path = stl::string(s_shader_path)+filename+".vs";
        stl::string vs_src;
        if(!ist::FileToString(vs_path, vs_src)) { istAssert(false); return false; }

        stl::string ps_path = stl::string(s_shader_path)+filename+".ps";
        stl::string ps_src;
        if(!ist::FileToString(ps_path, ps_src)) { istAssert(false); return false; }

        sh_desc.vs = dev->createVertexShader( VertexShaderDesc(vs_src.c_str(), vs_src.size()) );
        sh_desc.ps = dev->createPixelShader( PixelShaderDesc(ps_src.c_str(), ps_src.size()) );
        if(!sh_desc.vs || !sh_desc.ps) { return false; }
    }
#else // atm_enable_ShaderLiveEdit
    // todo: shader ファイルをアーカイブにまとめる
    static const char s_shader_path[] = "Resources/shader/";
    {
        stl::string vs_path = stl::string(s_shader_path)+filename+".vs";
        stl::string vs_src;
        if(!ist::FileToString(vs_path, vs_src)) { istAssert(false); return false; }

        stl::string ps_path = stl::string(s_shader_path)+filename+".ps";
        stl::string ps_src;
        if(!ist::FileToString(ps_path, ps_src)) { istAssert(false); return false; }

        sh_desc.vs = dev->createVertexShader( VertexShaderDesc(vs_src.c_str(), vs_src.size()) );
        sh_desc.ps = dev->createPixelShader( PixelShaderDesc(ps_src.c_str(), ps_src.size()) );
        if(!sh_desc.vs || !sh_desc.ps) { return false; }
    }
#endif // atm_enable_ShaderLiveEdit

    ShaderProgram *shader = dev->createShaderProgram(sh_desc);
    istSafeRelease(sh_desc.vs);
    istSafeRelease(sh_desc.ps);
    if(!shader) { istAssert(false); return false; }

    clearShaders();

    m_shader = shader;
    m_loc_renderstates = m_shader->getUniformBlockIndex("render_states");

    i3d::DeviceContext *dc = atmGetGLDeviceContext();
#define SetSampler(name, value) { GLint l=m_shader->getUniformLocation(name); if(l!=-1){ m_shader->setSampler(l, value); }}
    dc->setShader(m_shader);
    SetSampler("u_ColorBuffer",     GLSL_COLOR_BUFFER);
    SetSampler("u_NormalBuffer",    GLSL_NORMAL_BUFFER);
    SetSampler("u_PositionBuffer",  GLSL_POSITION_BUFFER);
    SetSampler("u_GlowBuffer",      GLSL_GLOW_BUFFER);
    SetSampler("u_BackBuffer",      GLSL_BACK_BUFFER);
    SetSampler("u_RandomBuffer",    GLSL_RANDOM_BUFFER);
    SetSampler("u_ParamBuffer",     GLSL_PARAM_BUFFER);
    dc->setShader(nullptr);
#undef SetSampler

    return true;
}

int32 AtomicShader::getUniformBlockIndex(const char *name) const
{
    return m_shader->getUniformBlockIndex(name);
}

void AtomicShader::setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, Buffer *buffer)
{
    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    dc->setUniformBuffer(uniformBlockIndex, uniformBindingIndex, buffer);
}

void AtomicShader::bind()
{
    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    assign(dc);
}

void AtomicShader::unbind()
{
    i3d::DeviceContext *ctx = atmGetGLDeviceContext();
    ctx->setShader(nullptr);
}

void AtomicShader::assign( i3d::DeviceContext *dc )
{
    dc->setShader(m_shader);
    dc->setUniformBuffer(m_loc_renderstates, GLSL_RENDERSTATE_BINDING, atmGetUniformBuffer(UBO_RENDERSTATES_3D));
}

#ifdef atm_enable_ShaderLiveEdit
bool AtomicShader::needsRecompile()
{
    static const char s_glsl_path[] = "shader/";
    if(m_shader) {
        Poco::File glsl((stl::string(s_glsl_path)+m_glsl_filename+".glsl").c_str());
        Poco::Timestamp lm = glsl.getLastModified();
        if(lm > m_timestamp) {
            return true;
        }
    }
    return false;
}

bool AtomicShader::recompile()
{
    return createShaders(m_glsl_filename.c_str());
}
#endif // atm_enable_ShaderLiveEdit

} // namespace atm