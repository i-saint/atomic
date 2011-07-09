#include "stdafx.h"
#include "../Base/Assert.h"
#include "../Math/SFMT/SFMT.h"
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

bool GenerateRandomTexture(Texture2D &tex, GLsizei width, GLsizei height, Texture2D::FORMAT format)
{
    static SFMT random;
    return GenerateRandomTexture(tex, width, height, format, random);
}

bool GenerateRandomTexture(Texture2D &tex, GLsizei width, GLsizei height, Texture2D::FORMAT format, SFMT& random)
{
    char *data = NULL;
    if(format==Texture2D::FMT_RGB_U8) {
        int data_size = width*height*3;
        data = new char[data_size];
        for(int i=0; i<data_size; ++i) {
            data[i] = random.genInt32();
        }
    }
    else if(format==Texture2D::FMT_RGBA_U8) {
        int data_size = width*height*4;
        data = new char[data_size];
        for(int i=0; i<data_size; ++i) {
            data[i] = random.genInt32();
        }
    }
    else if(format==Texture2D::FMT_RGB_F32) {
        int data_size = width*height*sizeof(float)*3;
        data = new char[data_size];
        float *w = (float*)data;
        for(int i=0; i<width*height*3; ++i) {
            w[i] = random.genFloat32();
        }
    }
    else if(format==Texture2D::FMT_RGBA_F32) {
        int data_size = width*height*sizeof(float)*4;
        data = new char[data_size];
        float *w = (float*)data;
        for(int i=0; i<width*height*4; ++i) {
            w[i] = random.genFloat32();
        }
    }
    else {
        IST_ASSERT("–¢ŽÀ‘•");
    }

    bool ret =  tex.initialize(width, height, format, data);
    delete[] data;
    return ret;
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
ColorNBuffer<NumColorBuffers>::ColorNBuffer()
: m_width(0)
, m_height(0)
{
    std::fill_n(m_owned, _countof(m_owned), (Texture2D*)NULL);
    std::fill_n(m_color, _countof(m_color), (Texture2D*)NULL);
}

template<size_t NumColorBuffers>
ColorNBuffer<NumColorBuffers>::~ColorNBuffer()
{
    for(size_t i=0; i<_countof(m_owned); ++i) {
        delete m_owned[i];
    }
}

template<size_t NumColorBuffers>
bool ColorNBuffer<NumColorBuffers>::initialize(GLsizei width, GLsizei height, FORMAT fmt)
{
    super::initialize();

    m_width = width;
    m_height = height;

    int num_owned = 0;
    for(size_t i=0; i<NumColorBuffers; ++i) {
        if(!m_color[i]) {
            Texture2D *color = new Texture2D();
            color->initialize(width, height, Texture2D::FORMAT(fmt));
            color->bind();
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            color->unbind();
            m_owned[num_owned++] = color;
            m_color[i] = color;
        }
        attachTexture(*m_color[i], FrameBufferObject::ATTACH(ATTACH_COLOR0+i));
    }
    return true;
}
template ColorNBuffer<1>;
template ColorNBuffer<2>;
template ColorNBuffer<3>;
template ColorNBuffer<4>;
template ColorNBuffer<5>;
template ColorNBuffer<6>;
template ColorNBuffer<7>;
template ColorNBuffer<8>;




DepthBuffer::DepthBuffer()
: m_width(0)
, m_height(0)
{
    m_owned = NULL;
    m_depth = NULL;
}

DepthBuffer::~DepthBuffer()
{
    delete m_owned;
}

bool DepthBuffer::initialize(GLsizei width, GLsizei height)
{
    super::initialize();

    m_width = width;
    m_height = height;

    if(!m_depth) {
        Texture2D *depth = new Texture2D();
        depth->initialize(width, height, Texture2D::FMT_DEPTH_F32);
        depth->bind();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        depth->unbind();
        m_owned = depth;
        m_depth = depth;
    }
    attachTexture(*m_depth, ATTACH_DEPTH);

    bind();
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    unbind();

    return true;
}




template<size_t NumColorBuffers>
ColorNDepthBuffer<NumColorBuffers>::ColorNDepthBuffer()
: m_width(0)
, m_height(0)
{
    std::fill_n(m_owned, _countof(m_owned), (Texture2D*)NULL);
    m_depth = NULL;
    std::fill_n(m_color, _countof(m_color), (Texture2D*)NULL);
}

template<size_t NumColorBuffers>
ColorNDepthBuffer<NumColorBuffers>::~ColorNDepthBuffer()
{
    for(size_t i=0; i<_countof(m_owned); ++i) {
        delete m_owned[i];
    }
}

template<size_t NumColorBuffers>
bool ColorNDepthBuffer<NumColorBuffers>::initialize(GLsizei width, GLsizei height, FORMAT fmt)
{
    super::initialize();

    m_width = width;
    m_height = height;

    int num_owned = 0;
    {
        if(!m_depth) {
            Texture2D *depth = new Texture2D();
            depth->initialize(width, height, Texture2D::FMT_DEPTH_F32);
            depth->bind();
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            depth->unbind();
            m_owned[num_owned++] = depth;
            m_depth = depth;
        }
        attachTexture(*m_depth, ATTACH_DEPTH);
    }
    for(size_t i=0; i<NumColorBuffers; ++i) {
        if(!m_color[i]) {
            Texture2D *color = new Texture2D();
            color->initialize(width, height, Texture2D::FORMAT(fmt));
            color->bind();
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            color->unbind();
            m_owned[num_owned++] = color;
            m_color[i] = color;
        }
        attachTexture(*m_color[i], FrameBufferObject::ATTACH(ATTACH_COLOR0+i));
    }
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
