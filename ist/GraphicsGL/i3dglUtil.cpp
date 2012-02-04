#include "stdafx.h"
#include "../Base.h"
#include "../Math.h"
#include "i3dglDevice.h"
#include "i3dglShader.h"
#include "i3dglUtil.h"
#include <string>
#include <sstream>
#include <fstream>

namespace ist {
namespace i3dgl {

Texture2D* CreateTexture2DFromFile(Device *dev, const char *filename)
{
    std::ifstream  st(filename, std::ios::binary);
    if(st.fail()) {
        istAssert("file not found %s", filename);
        return false;
    }
    return CreateTexture2DFromStream(dev, st);
}

Texture2D* CreateTexture2DFromStream(Device *dev, std::istream& st)
{
    istAssert("not implemented");
    return false;
}

Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format)
{
    static SFMT random;
    if(!random.isInitialized()) { random.initialize((uint32_t)::time(0)); }
    return GenerateRandomTexture(dev, size, format, random);
}

Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format, SFMT& random)
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

    Texture2DDesc desc = Texture2DDesc(format, size, 0, &buffer[0]);
    return dev->createTexture2D(desc);
}


template<class ShaderType> inline ShaderType* CreateShaderFromString(Device *dev, const std::string &source);

template<> inline VertexShader* CreateShaderFromString<VertexShader>(Device *dev, const std::string &source)
{
    VertexShaderDesc desc = VertexShaderDesc(source.c_str(), source.size());
    return dev->createVertexShader(desc);
}
template<> inline PixelShader* CreateShaderFromString<PixelShader>(Device *dev, const std::string &source)
{
    PixelShaderDesc desc = PixelShaderDesc(source.c_str(), source.size());
    return dev->createPixelShader(desc);
}
template<> inline GeometryShader* CreateShaderFromString<GeometryShader>(Device *dev, const std::string &source)
{
    GeometryShaderDesc desc = GeometryShaderDesc(source.c_str(), source.size());
    return dev->createGeometryShader(desc);
}

template<class ShaderType>
inline ShaderType* CreateShaderFromStream(Device *dev, std::istream& st)
{
    std::string source;
    std::ostringstream str_out;
    str_out << st.rdbuf();
    source = str_out.str();

    return CreateShaderFromString<ShaderType>(dev, source);
}

template<class ShaderType>
inline ShaderType* CreateShaderFromFile(Device *dev, const char *filename)
{
    std::ifstream  st(filename, std::ios::binary);
    if(st.fail()) {
        istAssert("file not found %s", filename);
        return NULL;
    }
    return CreateShaderFromStream<ShaderType>(dev, st);
}

VertexShader*   CreateVertexShaderFromFile(Device *dev, const char *filename)       { return CreateShaderFromFile<VertexShader>(dev, filename); }
GeometryShader* CreateGeometryShaderFromFile(Device *dev, const char *filename)     { return CreateShaderFromFile<GeometryShader>(dev, filename); }
PixelShader*    CreatePixelShaderFromFile(Device *dev, const char *filename)        { return CreateShaderFromFile<PixelShader>(dev, filename); }

VertexShader*   CreateVertexShaderFromStream(Device *dev, std::istream& st)         { return CreateShaderFromStream<VertexShader>(dev, st); }
GeometryShader* CreateGeometryShaderFromStream(Device *dev, std::istream& st)       { return CreateShaderFromStream<GeometryShader>(dev, st); }
PixelShader*    CreatePixelShaderFromStream(Device *dev, std::istream& st)          { return CreateShaderFromStream<PixelShader>(dev, st); }

VertexShader*   CreateVertexShaderFromString(Device *dev, const std::string &source)    { return CreateShaderFromString<VertexShader>(dev, source); }
GeometryShader* CreateGeometryShaderFromString(Device *dev, const std::string &source)  { return CreateShaderFromString<GeometryShader>(dev, source); }
PixelShader*    CreatePixelShaderFromString(Device *dev, const std::string &source)     { return CreateShaderFromString<PixelShader>(dev, source); }



Texture2D* CreateRenderBufferTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT color_format, uint32 mipmaps)
{
    Texture2DDesc desc = Texture2DDesc(color_format, size, mipmaps);
    return dev->createTexture2D(desc);
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, uint32 mipmaps)
{
    I3D_COLOR_FORMAT color_formats[I3D_MAX_RENDER_TARGETS];
    stl::fill_n(color_formats, num_color_buffers, color_format);
    return CreateRenderTarget(dev, num_color_buffers, size, color_formats, mipmaps);
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, uint32 mipmaps)
{
    Texture2D *rb[I3D_MAX_RENDER_TARGETS];
    RenderTarget *rt = dev->createRenderTarget();
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i] = CreateRenderBufferTexture(dev, size, color_formats[i], mipmaps);
    }
    rt->setRenderBuffers(rb, num_color_buffers, NULL);
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i]->release();
    }
    return rt;
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, I3D_COLOR_FORMAT depthstencil_format, uint32 mipmaps)
{
    I3D_COLOR_FORMAT color_formats[I3D_MAX_RENDER_TARGETS];
    stl::fill_n(color_formats, num_color_buffers, color_format);
    return CreateRenderTarget(dev, num_color_buffers, size, color_formats, depthstencil_format, mipmaps);
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, I3D_COLOR_FORMAT depthstencil_format, uint32 mipmaps)
{
    Texture2D *rb[I3D_MAX_RENDER_TARGETS];
    Texture2D *ds;
    RenderTarget *rt = dev->createRenderTarget();
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i] = CreateRenderBufferTexture(dev, size, color_formats[i], mipmaps);
    }
    {
        ds = CreateRenderBufferTexture(dev, size, depthstencil_format, mipmaps);
    }
    rt->setRenderBuffers(rb, num_color_buffers, ds);
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i]->release();
    }
    {
        ds->release();
    }
    return rt;
}

Buffer* CreateVertexBuffer( Device *dev, uint32 size, I3D_USAGE usage, void *data/*=NULL*/ )
{
    BufferDesc desc(I3D_VERTEX_BUFFER, usage, size, data);
    return dev->createBuffer(desc);
}

Buffer* CreateIndexBuffer( Device *dev, uint32 size, I3D_USAGE usage, void *data/*=NULL*/ )
{
    BufferDesc desc(I3D_INDEX_BUFFER, usage, size, data);
    return dev->createBuffer(desc);
}

Buffer* CreateUniformBuffer( Device *dev, uint32 size, I3D_USAGE usage, void *data/*=NULL*/ )
{
    BufferDesc desc(I3D_UNIFORM_BUFFER, usage, size, data);
    return dev->createBuffer(desc);
}


} // namespace i3d
} // namespace ist
