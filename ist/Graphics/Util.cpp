#include "stdafx.h"
#include "../Base/Assert.h"
#include "GraphicsAssert.h"
#include "ShaderObject.h"
#include "Util.h"
#include <string>
#include <sstream>

namespace ist {
namespace graphics {

bool CreateTexture2DFromFile(Texture2D& tex, const char *filename)
{
    std::ifstream  st(filename, std::ios::binary);
    if(st.fail()) {
        IST_ASSERT("file not found %s", filename);
        return false;
    }
    return CreateTexture2DFromStream(tex, st);
}

bool CreateTexture2DFromStream(Texture2D& tex, std::istream& st)
{
    IST_ASSERT("not implemented");
    return false;
}



template<class ShaderType>
bool CreateShaderFromFile(ShaderType& sh, const char *filename)
{
    std::ifstream  st(filename, std::ios::binary);
    if(st.fail()) {
        IST_ASSERT("file not found %s", filename);
        return false;
    }
    return CreateShaderFromStream<ShaderType>(sh, st);
}

template<class ShaderType>
bool CreateShaderFromStream(ShaderType& sh, std::istream& st)
{
    std::string source;
    std::ostringstream str_out;
    str_out << st.rdbuf();
    source = str_out.str();

    return sh.initialize(source.c_str(), source.size());
}

bool CreateVertexShaderFromFile(VertexShader& sh, const char *filename)     { return CreateShaderFromFile<VertexShader>(sh, filename); }
bool CreateGeometryShaderFromFile(GeometryShader& sh, const char *filename) { return CreateShaderFromFile<GeometryShader>(sh, filename); }
bool CreateFragmentShaderFromFile(FragmentShader& sh, const char *filename) { return CreateShaderFromFile<FragmentShader>(sh, filename); }
bool CreateVertexShaderFromStream(VertexShader& sh, std::istream& st)       { return CreateShaderFromStream<VertexShader>(sh, st); }
bool CreateGeometryShaderFromStream(GeometryShader& sh, std::istream& st)   { return CreateShaderFromStream<GeometryShader>(sh, st); }
bool CreateFragmentShaderFromStream(FragmentShader& sh, std::istream& st)   { return CreateShaderFromStream<FragmentShader>(sh, st); }



template<size_t NumColorBuffers>
bool ColorNBuffer<NumColorBuffers>::initialize(GLsizei width, GLsizei height)
{
    for(size_t i=0; i<NumColorBuffers; ++i) {
        m_color[i].initialize(width, height, Texture2D::FMT_RGBA_I8);
        m_color[i].bind();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        m_color[i].unbind();
        attachTexture(m_color[i], FrameBufferObject::ATTACH(ATTACH_COLOR0+i));
    }
}
template ColorNBuffer<1>;
template ColorNBuffer<2>;
template ColorNBuffer<3>;
template ColorNBuffer<4>;
template ColorNBuffer<5>;
template ColorNBuffer<6>;
template ColorNBuffer<7>;
template ColorNBuffer<8>;


bool DepthBuffer::initialize(GLsizei width, GLsizei height)
{
    m_depth.initialize(width, height, Texture2D::FMT_DEPTH_F32);
    m_depth.bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    m_depth.unbind();

    attachTexture(m_depth, ATTACH_DEPTH);
    bind();
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    unbind();
    return true;
}


template<size_t NumColorBuffers>
bool ColorNDepthBuffer<NumColorBuffers>::initialize(GLsizei width, GLsizei height)
{
    for(size_t i=0; i<NumColorBuffers; ++i) {
        m_color[i].initialize(width, height, Texture2D::FMT_RGBA_I8);
        m_color[i].bind();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        m_color[i].unbind();
        attachTexture(m_color[i], FrameBufferObject::ATTACH(ATTACH_COLOR0+i));
    }

    m_depth.initialize(width, height, RenderBuffer::FMT_DEPTH_F32);
    attachRenderBuffer(m_depth, ATTACH_DEPTH);
    return true;
}
template ColorNDepthBuffer<1>;
template ColorNDepthBuffer<2>;
template ColorNDepthBuffer<3>;
template ColorNDepthBuffer<4>;
template ColorNDepthBuffer<5>;
template ColorNDepthBuffer<6>;
template ColorNDepthBuffer<7>;
template ColorNDepthBuffer<8>;


} // namespace graphics
} // namespace ist
