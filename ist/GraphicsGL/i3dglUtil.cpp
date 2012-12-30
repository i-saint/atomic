#include "istPCH.h"
#ifdef ist_with_OpenGL
#include "ist/Base.h"
#include "ist/Math.h"
#include "ist/GraphicsCommon/Image.h"
#include "i3dglDevice.h"
#include "i3dglShader.h"
#include "i3dglUtil.h"
#include <string>
#include <sstream>
#include <fstream>
#include <ctime>

namespace ist {
namespace i3dgl {

Texture2D* CreateTexture2DFromFile(Device *dev, const char *filename, I3D_COLOR_FORMAT format)
{
    Image img;
    if(!img.load(filename)) {
        istPrint("file load failed: %s\n", filename);
        return NULL;
    }
    return CreateTexture2DFromImage(dev, img, format);
}

Texture2D* CreateTexture2DFromStream(Device *dev, IBinaryStream &st, I3D_COLOR_FORMAT format)
{
    Image img;
    if(!img.load(st)) {
        istPrint("file load failed\n");
        return NULL;
    }
    return CreateTexture2DFromImage(dev, img, format);
}

Texture2D* CreateTexture2DFromImage(Device *dev, Image &img, I3D_COLOR_FORMAT format)
{
    if(format==I3D_COLOR_UNKNOWN) {
        switch(img.getFormat()) {
        case IF_R8U:        format=I3D_R8; break;
        case IF_R8I:        format=I3D_R8S; break;
        case IF_R32F:       format=I3D_R32F; break;
        case IF_RG8U:       format=I3D_RG8; break;
        case IF_RG8I:       format=I3D_RG8S; break;
        case IF_RG32F:      format=I3D_RG32F; break;
        case IF_RGB8U:      format=I3D_RGB8; break;
        case IF_RGB8I:      format=I3D_RGB8S; break;
        case IF_RGB32F:     format=I3D_RGB32F; break;
        case IF_RGBA8U:     format=I3D_RGBA8; break;
        case IF_RGBA8I:     format=I3D_RGBA8S; break;
        case IF_RGBA32F:    format=I3D_RGBA32F; break;
        case IF_RGBA_DXT1:  format=I3D_RGBA_DXT1; break;
        case IF_RGBA_DXT3:  format=I3D_RGBA_DXT3; break;
        case IF_RGBA_DXT5:  format=I3D_RGBA_DXT5; break;
        }
    }
    Texture2DDesc desc(format, uvec2(img.width(), img.height()), 0, img.data(), img.size());
    return dev->createTexture2D(desc);
}


Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format)
{
    static SFMT random;
    if(!random.isInitialized()) { random.initialize((uint32_t)::time(0)); }
    return GenerateRandomTexture(dev, size, format, random);
}

Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format, SFMT& random)
{
    stl::string buffer;
    if(format==I3D_RGB8) {
        uint32 data_size = size.x*size.y*3;
        buffer.resize(data_size);
        for(uint32 i=0; i<data_size; ++i) {
            buffer[i] = random.genInt32();
        }
    }
    else if(format==I3D_RGBA8) {
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
        istAssert(false, "not implemented");
    }

    Texture2DDesc desc = Texture2DDesc(format, size, 0, &buffer[0]);
    return dev->createTexture2D(desc);
}


template<class ShaderType> inline ShaderType* CreateShaderFromString(Device *dev, const stl::string &source);

template<> inline VertexShader* CreateShaderFromString<VertexShader>(Device *dev, const stl::string &source)
{
    VertexShaderDesc desc = VertexShaderDesc(source.c_str(), source.size());
    return dev->createVertexShader(desc);
}
template<> inline PixelShader* CreateShaderFromString<PixelShader>(Device *dev, const stl::string &source)
{
    PixelShaderDesc desc = PixelShaderDesc(source.c_str(), source.size());
    return dev->createPixelShader(desc);
}
template<> inline GeometryShader* CreateShaderFromString<GeometryShader>(Device *dev, const stl::string &source)
{
    GeometryShaderDesc desc = GeometryShaderDesc(source.c_str(), source.size());
    return dev->createGeometryShader(desc);
}

template<class ShaderType>
inline ShaderType* CreateShaderFromStream(Device *dev, IBinaryStream &st)
{
    stl::string src;
    char tmp[1024];
    uint64 read_size;
    while( (read_size=st.read(tmp, _countof(tmp))) > 0 ) {
        src.insert(src.end(), tmp, tmp+read_size);
    }
    return CreateShaderFromString<ShaderType>(dev, src);
}

template<class ShaderType>
inline ShaderType* CreateShaderFromFile(Device *dev, const char *filename)
{
    FileStream  st(filename, "rb");
    if(!st.isOpened()) {
        istAssert("file not found %s", filename);
        return NULL;
    }
    return CreateShaderFromStream<ShaderType>(dev, st);
}

VertexShader*   CreateVertexShaderFromFile(Device *dev, const char *filename)       { return CreateShaderFromFile<VertexShader>(dev, filename); }
GeometryShader* CreateGeometryShaderFromFile(Device *dev, const char *filename)     { return CreateShaderFromFile<GeometryShader>(dev, filename); }
PixelShader*    CreatePixelShaderFromFile(Device *dev, const char *filename)        { return CreateShaderFromFile<PixelShader>(dev, filename); }

VertexShader*   CreateVertexShaderFromStream(Device *dev, IBinaryStream &st)        { return CreateShaderFromStream<VertexShader>(dev, st); }
GeometryShader* CreateGeometryShaderFromStream(Device *dev, IBinaryStream &st)      { return CreateShaderFromStream<GeometryShader>(dev, st); }
PixelShader*    CreatePixelShaderFromStream(Device *dev, IBinaryStream &st)         { return CreateShaderFromStream<PixelShader>(dev, st); }

VertexShader*   CreateVertexShaderFromString(Device *dev, const stl::string &source)    { return CreateShaderFromString<VertexShader>(dev, source); }
GeometryShader* CreateGeometryShaderFromString(Device *dev, const stl::string &source)  { return CreateShaderFromString<GeometryShader>(dev, source); }
PixelShader*    CreatePixelShaderFromString(Device *dev, const stl::string &source)     { return CreateShaderFromString<PixelShader>(dev, source); }



Texture2D* CreateRenderBufferTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT color_format, uint32 level_color)
{
    Texture2DDesc desc = Texture2DDesc(color_format, size, level_color);
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
    I3D_COLOR_FORMAT *color_formats, uint32 level_color)
{
    Texture2D *rb[I3D_MAX_RENDER_TARGETS];
    RenderTarget *rt = dev->createRenderTarget();
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i] = CreateRenderBufferTexture(dev, size, color_formats[i], level_color);
    }
    rt->setRenderBuffers(rb, num_color_buffers, NULL);
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i]->release();
    }
    return rt;
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, I3D_COLOR_FORMAT depthstencil_format, uint32 level_color, uint32 level_depth)
{
    I3D_COLOR_FORMAT color_formats[I3D_MAX_RENDER_TARGETS];
    stl::fill_n(color_formats, num_color_buffers, color_format);
    return CreateRenderTarget(dev, num_color_buffers, size, color_formats, depthstencil_format, level_color, level_depth);
}

RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, I3D_COLOR_FORMAT depthstencil_format, uint32 level_color, uint32 level_depth)
{
    Texture2D *rb[I3D_MAX_RENDER_TARGETS];
    Texture2D *ds;
    RenderTarget *rt = dev->createRenderTarget();
    for(uint32 i=0; i<num_color_buffers; ++i) {
        rb[i] = CreateRenderBufferTexture(dev, size, color_formats[i], level_color);
    }
    {
        ds = CreateRenderBufferTexture(dev, size, depthstencil_format, level_depth);
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

void EnableVSync( bool v )
{
    wglSwapIntervalEXT(v);
}


} // namespace i3d
} // namespace ist
#endif // ist_with_OpenGL
