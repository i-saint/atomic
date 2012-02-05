#ifndef __ist_i3dgl_Types__
#define __ist_i3dgl_Types__

#include "ist/Base/New.h"

namespace ist {
namespace i3dgl {

enum I3D_DEVICE_RESOURCE_TYPE {
    I3D_TEXTURE,
    I3D_BUFFER,
    I3D_VERTEX_ARRAY,
    I3D_SHADER,
    I3D_FRAME_BUFFER,
    I3D_RENDER_BUFFER,
};

enum I3D_TOPOLOGY {
    I3D_POINTS      = GL_POINTS,
    I3D_LINES       = GL_LINES,
    I3D_TRIANGLES   = GL_TRIANGLES,
    I3D_QUADS       = GL_QUADS,
};

enum I3D_COLOR_FORMAT
{
    I3D_R8U,
    I3D_R16F,
    I3D_R32F,
    I3D_RG8U,
    I3D_RG16F,
    I3D_RG32F,
    I3D_RGB8U,
    I3D_RGB16F,
    I3D_RGB32F,
    I3D_RGBA8U,
    I3D_RGBA16F,
    I3D_RGBA32F,
    I3D_DEPTH32F,
    I3D_DEPTH24_STENCIL8,
    I3D_DEPTH32F_STENCIL8,
};

enum I3D_USAGE
{
    I3D_USAGE_STATIC    = GL_STATIC_DRAW,
    I3D_USAGE_DYNAMIC   = GL_DYNAMIC_DRAW,
    I3D_USAGE_STREAM    = GL_STREAM_DRAW,
};

enum I3D_MAP_MODE
{
    I3D_MAP_READ        = GL_READ_ONLY,
    I3D_MAP_WRITE       = GL_WRITE_ONLY,
    I3D_MAP_READWRITE   = GL_READ_WRITE,
};

enum I3D_TYPE
{
    I3D_BYTE    = GL_BYTE,
    I3D_UBYTE   = GL_UNSIGNED_BYTE,
    I3D_SHORT   = GL_SHORT,
    I3D_USHORT  = GL_UNSIGNED_SHORT,
    I3D_INT     = GL_INT,
    I3D_UINT    = GL_UNSIGNED_INT,
    I3D_HALF    = GL_HALF_FLOAT,
    I3D_FLOAT   = GL_FLOAT,
    I3D_DOUBLE  = GL_DOUBLE,
};

enum I3D_BUFFER_TYPE
{
    I3D_VERTEX_BUFFER       = GL_ARRAY_BUFFER,
    I3D_INDEX_BUFFER        = GL_ELEMENT_ARRAY_BUFFER,
    I3D_UNIFORM_BUFFER      = GL_UNIFORM_BUFFER,
    I3D_PIXEL_PACK_BUFFER   = GL_PIXEL_PACK_BUFFER,
    I3D_PIXEL_UNPACK_BUFFER = GL_PIXEL_UNPACK_BUFFER,
};

enum I3D_CONSTANTS
{
    I3D_MAX_RENDER_TARGETS  = 8,
    I3D_MAX_VERTEX_BUFFERS  = 8,
    I3D_MAX_VERTEX_DESCS    = 16,
};

enum I3D_TEXTURE_CLAMP
{
    I3D_REPEAT          = GL_REPEAT,
    I3D_MIRRORED_REPEAT = GL_MIRRORED_REPEAT,
    I3D_CLAMP_TO_EDGE   = GL_CLAMP_TO_EDGE,
    I3D_CLAMP_TO_BORDER = GL_CLAMP_TO_BORDER,
};

enum I3D_TEXTURE_FILTER
{
    I3D_NEAREST                 = GL_NEAREST,
    I3D_LINEAR                  = GL_LINEAR,
    I3D_NEAREST_MIPMAP_NEAREST  = GL_NEAREST_MIPMAP_NEAREST,
    I3D_NEAREST_MIPMAP_LINEAR   = GL_NEAREST_MIPMAP_LINEAR,
    I3D_LINEAR_MIPMAP_NEAREST   = GL_LINEAR_MIPMAP_NEAREST,
    I3D_LINEAR_MIPMAP_LINEAR    = GL_LINEAR_MIPMAP_LINEAR,
};


typedef uint32 ResourceHandle;
class Device;
class DeviceContext;
class DeviceResource;
class Buffer;
class VertexArray;
class Texture1D;
class Texture2D;
class Texture3D;
typedef Texture2D RenderBuffer;
class RenderTarget;
class VertexShader;
class PixelShader;
class GeometryShader;
class ShaderProgtam;


struct VertexDesc
{
    GLuint location;        // shader value location
    I3D_TYPE type;          // value type
    GLuint num_elements;    // must be 1,2,3,4
    GLuint offset;
    bool normalize;
    GLuint divisor; // 0: per vertex, other: per n instance
};

struct BufferDesc
{
    I3D_BUFFER_TYPE type;
    I3D_USAGE usage;
    uint32 size;
    void *data;

    // data は NULL でもよく、その場合メモリ確保だけが行われる。
    explicit BufferDesc(I3D_BUFFER_TYPE _type, I3D_USAGE _usage=I3D_USAGE_DYNAMIC, uint32 _size=0, void *_data=NULL)
        : type(_type)
        , usage(_usage)
        , size(_size)
        , data(_data)
    {}
};

struct SamplerDesc
{
    I3D_TEXTURE_CLAMP wrap_s;
    I3D_TEXTURE_CLAMP wrap_t;
    I3D_TEXTURE_CLAMP wrap_r;
    I3D_TEXTURE_FILTER filter_min;
    I3D_TEXTURE_FILTER filter_mag;

    explicit SamplerDesc(
        I3D_TEXTURE_CLAMP _s=I3D_CLAMP_TO_EDGE, I3D_TEXTURE_CLAMP _t=I3D_CLAMP_TO_EDGE, I3D_TEXTURE_CLAMP _r=I3D_CLAMP_TO_EDGE,
        I3D_TEXTURE_FILTER _min=I3D_NEAREST, I3D_TEXTURE_FILTER _mag=I3D_NEAREST)
        : wrap_s(_s), wrap_t(_t), wrap_r(_r)
        , filter_min(_min), filter_mag(_mag)
    {}
};

struct Texture1DDesc
{
    I3D_COLOR_FORMAT format;
    uint32 size;
    uint32 mipmap;
    void *data;

    explicit Texture1DDesc(I3D_COLOR_FORMAT _format=I3D_RGBA8U, uint32 _size=0, uint32 _mipmap=0, void *_data=NULL)
        : format(_format)
        , size(_size)
        , mipmap(_mipmap)
        , data(_data)
    {}
};

struct Texture2DDesc
{
    I3D_COLOR_FORMAT format;
    uvec2 size;
    uint32 mipmap;
    void *data;

    explicit Texture2DDesc(I3D_COLOR_FORMAT _format=I3D_RGBA8U, uvec2 _size=uvec2(0, 0), uint32 _mipmap=0, void *_data=NULL)
        : format(_format)
        , size(_size)
        , mipmap(_mipmap)
        , data(_data)
    {}
};

struct Texture3DDesc
{
    I3D_COLOR_FORMAT format;
    uvec3 size;
    uint32 mipmap;
    void *data;

    explicit Texture3DDesc(I3D_COLOR_FORMAT _format=I3D_RGBA8U, uvec3 _size=uvec3(0, 0, 0), uint32 _mipmap=0, void *_data=NULL)
        : format(_format)
        , size(_size)
        , mipmap(_mipmap)
        , data(_data)
    {}
};

struct ShaderDesc
{
    const char *source;
    uint32 source_len;

    explicit ShaderDesc(const char *s=NULL, uint32 l=0) : source(s), source_len(l) {}
};
typedef ShaderDesc VertexShaderDesc;
typedef ShaderDesc GeometryShaderDesc;
typedef ShaderDesc PixelShaderDesc;

struct ShaderProgramDesc
{
    VertexShader    *vsh;
    PixelShader     *psh;
    GeometryShader  *gsh;

    explicit ShaderProgramDesc(VertexShader *v=NULL, PixelShader *p=NULL, GeometryShader *g=NULL)
        : vsh(v), psh(p), gsh(g)
    {}
};


} // namespace i3d
} // namespace ist

#define I3DGL_DECLARE_DEVICE_RESOURCE(classname) \
private:\
    template<class T> friend T* ::call_destructor(T *v);\
    friend class Device;\
    friend class DeviceContext;

#endif // __ist_i3dgl_Types__
