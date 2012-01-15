#include "stdafx.h"
#include "../Base.h"
#include "../Math.h"
#include <D3D11.h>
#include <D3DX11.h>
#include <string>
#include <sstream>
#include <fstream>
#include "i3ddx11Device.h"
#include "i3ddx11Shader.h"
#include "i3ddx11Util.h"

namespace ist {
namespace i3ddx11 {

bool CreateTexture2DFromFile(Texture2D& tex, const char *filename)
{
    std::ifstream  st(filename, std::ios::binary);
    if(st.fail()) {
        istAssert("file not found %s", filename);
        return false;
    }
    return CreateTexture2DFromStream(tex, st);
}

bool CreateTexture2DFromStream(Texture2D& tex, std::istream& st)
{
    istAssert("not implemented");
    return false;
}

bool GenerateRandomTexture(Texture2D &tex, const uvec2 &size, I3D_COLOR_FORMAT format)
{
    static SFMT random;
    if(!random.isInitialized()) { random.initialize((uint32_t)::time(0)); }
    return GenerateRandomTexture(tex, size, format, random);
}

bool GenerateRandomTexture(Texture2D &tex, const uvec2 &size, I3D_COLOR_FORMAT format, SFMT& random)
{
    std::string buffer;
    if(format==I3D_RGB8U) {
        uint32 data_size = size.x*size.y*3;
        buffer.resize(data_size);
        for(uint32 i=0; i<data_size; ++i) {
            buffer[i] = random.genInt32();
        }
    }
    else if(format==I3D_RGBA8U) {
        uint32 data_size = size.x*size.y*4;
        buffer.resize(data_size);
        for(uint32 i=0; i<data_size; ++i) {
            buffer[i] = random.genInt32();
        }
    }
    else if(format==I3D_RGB32F) {
        uint32 data_size = size.x*size.y*sizeof(float)*3;
        buffer.resize(data_size);
        float32 *w = (float*)&buffer[0];
        for(uint32 i=0; i<size.x*size.y*3; ++i) {
            w[i] = random.genFloat32();
        }
    }
    else if(format==I3D_RGBA32F) {
        uint32 data_size = size.x*size.y*sizeof(float)*4;
        buffer.resize(data_size);
        float32 *w = (float*)&buffer[0];
        for(uint32 i=0; i<size.x*size.y*4; ++i) {
            w[i] = random.genFloat32();
        }
    }
    else {
        istAssert("–¢ŽÀ‘•");
    }

    bool ret =  tex.allocate(size, format, &buffer[0]);
    return ret;
}

//
//template<class ShaderType>
//inline bool CreateShaderFromFile(ShaderType& sh, const char *filename)
//{
//    std::ifstream  st(filename, std::ios::binary);
//    if(st.fail()) {
//        istAssert("file not found %s", filename);
//        return false;
//    }
//    return CreateShaderFromStream<ShaderType>(sh, st);
//}
//
//template<class ShaderType>
//inline bool CreateShaderFromStream(ShaderType& sh, std::istream& st)
//{
//    std::string source;
//    std::ostringstream str_out;
//    str_out << st.rdbuf();
//    source = str_out.str();
//
//    return sh.compile(source.c_str(), source.size());
//}
//
//template<class ShaderType>
//inline bool CreateShaderFromString(ShaderType& sh, const char* source)
//{
//    return sh.compile(source, strlen(source));
//}
//
//bool CreateVertexShaderFromFile(VertexShader& sh, const char *filename)     { return CreateShaderFromFile<VertexShader>(sh, filename); }
//bool CreateGeometryShaderFromFile(GeometryShader& sh, const char *filename) { return CreateShaderFromFile<GeometryShader>(sh, filename); }
//bool CreateFragmentShaderFromFile(PixelShader& sh, const char *filename)    { return CreateShaderFromFile<PixelShader>(sh, filename); }
//bool CreateVertexShaderFromStream(VertexShader& sh, std::istream& st)       { return CreateShaderFromStream<VertexShader>(sh, st); }
//bool CreateGeometryShaderFromStream(GeometryShader& sh, std::istream& st)   { return CreateShaderFromStream<GeometryShader>(sh, st); }
//bool CreateFragmentShaderFromStream(PixelShader& sh, std::istream& st)      { return CreateShaderFromStream<PixelShader>(sh, st); }
//bool CreateVertexShaderFromString(VertexShader& sh, const char* source)     { return CreateShaderFromString<VertexShader>(sh, source); }
//bool CreateGeometryShaderFromString(GeometryShader& sh, const char* source) { return CreateShaderFromString<GeometryShader>(sh, source); }
//bool CreateFragmentShaderFromString(PixelShader& sh, const char* source)    { return CreateShaderFromString<PixelShader>(sh, source); }



Texture2D* CreateRenderBufferTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT color_format)
{
    Texture2D *r = dev->createTexture2D();
    r->allocate(size, color_format);
    return r;
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format)
{
    I3D_COLOR_FORMAT color_formats[I3D_MAX_RENDER_TARGETS];
    std::fill_n(color_formats, num_color_buffers, color_format);
    return CreateRenderTarget(dev, num_color_buffers, size, color_formats);
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats)
{
    Texture2D *rb[I3D_MAX_RENDER_TARGETS];
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i] = CreateRenderBufferTexture(dev, size, color_formats[i]);
    }
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i]->release();
    }
    return NULL;
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, I3D_COLOR_FORMAT depthstencil_format)
{
    I3D_COLOR_FORMAT color_formats[I3D_MAX_RENDER_TARGETS];
    std::fill_n(color_formats, num_color_buffers, color_format);
    return CreateRenderTarget(dev, num_color_buffers, size, color_formats, depthstencil_format);
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, I3D_COLOR_FORMAT depthstencil_format)
{
    Texture2D *rb[I3D_MAX_RENDER_TARGETS];
    Texture2D *ds;
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i] = CreateRenderBufferTexture(dev, size, color_formats[i]);
    }
    {
        ds = CreateRenderBufferTexture(dev, size, depthstencil_format);
    }
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i]->release();
    }
    {
        ds->release();
    }
    return NULL;
}


} // namespace i3ddx11
} // namespace ist
