#ifndef __ist_i3dgl_Types__
#define __ist_i3dgl_Types__

#include "ist/Base/New.h"
#include "ist/Base/SharedObject.h"
#include "ist/Base/NonCopyable.h"

namespace ist {
namespace i3dgl {

enum I3D_CONSTANTS
{
    I3D_MAX_RENDER_TARGETS  = 8,
    I3D_MAX_VERTEX_BUFFERS  = 8,
    I3D_MAX_VERTEX_DESCS    = 32,
    I3D_MAX_TEXTURE_SLOTS   = 16,
};


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
    I3D_COLOR_UNKNOWN,

                //     CPU  ->    GPU
    I3D_R8,     //    0-255 ->  0.0f-1.0f
    I3D_R8S,    // -128-127 -> -1.0f-1.0f
    I3D_R8U,    //    0-255 ->     0-255
    I3D_R8I,    // -128-127 ->  -128-127
    I3D_R16F,
    I3D_R32F,
    I3D_RG8,
    I3D_RG8S,
    I3D_RG8U,
    I3D_RG8I,
    I3D_RG16F,
    I3D_RG32F,
    I3D_RGB8,
    I3D_RGB8S,
    I3D_RGB8U,
    I3D_RGB8I,
    I3D_RGB16F,
    I3D_RGB32F,
    I3D_RGBA8,
    I3D_RGBA8S,
    I3D_RGBA8U,
    I3D_RGBA8I,
    I3D_RGBA16F,
    I3D_RGBA32F,
    I3D_DEPTH16F,
    I3D_DEPTH32F,
    I3D_DEPTH24_STENCIL8,
    I3D_DEPTH32F_STENCIL8,
    I3D_RGB_DXT1,
    I3D_SRGB_DXT1,
    I3D_RGBA_DXT1,
    I3D_SRGBA_DXT1,
    I3D_RGBA_DXT3,
    I3D_SRGBA_DXT3,
    I3D_RGBA_DXT5,
    I3D_SRGBA_DXT5,
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

enum I3D_BLEND_EQUATION
{
    I3D_BLEND_ADD               = GL_FUNC_ADD,
    I3D_BLEND_SUBTRACT          = GL_FUNC_SUBTRACT,
    I3D_BLEND_REVERSE_SUBTRACT  = GL_FUNC_REVERSE_SUBTRACT,
    I3D_BLEND_MIN               = GL_MIN,
    I3D_BLEND_MAX               = GL_MAX,
};

enum I3D_BLEND_FUNC
{
    I3D_BLEND_ZERO              = GL_ZERO,
    I3D_BLEND_ONE               = GL_ONE,
    I3D_BLEND_SRC_COLOR         = GL_SRC_COLOR,
    I3D_BLEND_SRC_ALPHA         = GL_SRC_ALPHA,
    I3D_BLEND_INV_SRC_COLOR     = GL_ONE_MINUS_SRC_COLOR,
    I3D_BLEND_INV_SRC_ALPHA     = GL_ONE_MINUS_SRC_ALPHA,
    I3D_BLEND_DST_COLOR         = GL_DST_COLOR,
    I3D_BLEND_DST_ALPHA         = GL_DST_ALPHA,
    I3D_BLEND_INV_DST_COLOR     = GL_ONE_MINUS_DST_COLOR,
    I3D_BLEND_INV_DST_ALPHA     = GL_ONE_MINUS_DST_ALPHA,
};

enum I3D_DEPTH_FUNC
{
    I3D_DEPTH_NEVER         = GL_NEVER,
    I3D_DEPTH_ALWAYS        = GL_ALWAYS,
    I3D_DEPTH_LESS          = GL_LESS,
    I3D_DEPTH_LESS_EQUAL    = GL_LEQUAL,
    I3D_DEPTH_GREATER       = GL_GREATER,
    I3D_DEPTH_GREATER_EQUAL = GL_GEQUAL,
    I3D_DEPTH_EQUAL         = GL_EQUAL,
    I3D_DEPTH_NOT_EQUAL     = GL_NOTEQUAL,
};

enum I3D_STENCIL_FUNC
{
    I3D_STENCIL_NEVER           = GL_NEVER,
    I3D_STENCIL_ALWAYS          = GL_ALWAYS,
    I3D_STENCIL_LESS            = GL_LESS,
    I3D_STENCIL_LESS_EQUAL      = GL_LEQUAL,
    I3D_STENCIL_GREATER         = GL_GREATER,
    I3D_STENCIL_GREATER_EQUAL   = GL_GEQUAL,
    I3D_STENCIL_EQUAL           = GL_EQUAL,
    I3D_STENCIL_NOT_EQUAL       = GL_NOTEQUAL,
};

enum I3D_STENCIL_OP
{
    I3D_STENCIL_KEEP            = GL_KEEP,
    I3D_STENCIL_ZERO            = GL_ZERO,
    I3D_STENCIL_REPLACE         = GL_REPLACE,
    I3D_STENCIL_INCREMENT       = GL_INCR,
    I3D_STENCIL_INCREMENT_WRAP  = GL_INCR_WRAP,
    I3D_STENCIL_DECREMENT       = GL_DECR,
    I3D_STENCIL_DECREMENT_WRAP  = GL_DECR_WRAP,
    I3D_STENCIL_INVERT          = GL_INVERT,
};


typedef uint32 ResourceHandle;
class Device;
class DeviceContext;
class DeviceResource;
class Buffer;
class VertexArray;
class Sampler;
class Texture;
class Texture1D;
class Texture2D;
class Texture3D;
typedef Texture2D RenderBuffer;
class RenderTarget;
class VertexShader;
class PixelShader;
class GeometryShader;
class ShaderProgtam;
class BlendState;
class DepthStencilState;


struct istInterModule VertexDesc
{
    GLuint location;        // shader value location
    I3D_TYPE type;          // value type
    GLuint num_elements;    // must be 1,2,3,4
    GLuint offset;
    bool normalize;
    GLuint divisor; // 0: per vertex, other: per n instance
};

struct istInterModule BufferDesc
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

struct istInterModule SamplerDesc
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

struct istInterModule Texture1DDesc
{
    I3D_COLOR_FORMAT format;
    uint32 size;
    uint32 mipmap;
    void *data;
    uint32 data_size; // 圧縮フォーマットの場合のみ使用

    explicit Texture1DDesc(I3D_COLOR_FORMAT _format=I3D_RGBA8, uint32 _size=0, uint32 _mipmap=0, void *_data=NULL, size_t _data_size=0)
        : format(_format)
        , size(_size)
        , mipmap(_mipmap)
        , data(_data)
        , data_size(_data_size)
    {}
};

struct istInterModule Texture2DDesc
{
    I3D_COLOR_FORMAT format;
    uvec2 size;
    uint32 mipmap;
    void *data;
    uint32 data_size; // 圧縮フォーマットの場合のみ使用

    explicit Texture2DDesc(I3D_COLOR_FORMAT _format=I3D_RGBA8, uvec2 _size=uvec2(0, 0), uint32 _mipmap=0, void *_data=NULL, size_t _data_size=0)
        : format(_format)
        , size(_size)
        , mipmap(_mipmap)
        , data(_data)
        , data_size(_data_size)
    {}
};

struct istInterModule Texture3DDesc
{
    I3D_COLOR_FORMAT format;
    uvec3 size;
    uint32 mipmap;
    void *data;
    uint32 data_size; // 圧縮フォーマットの場合のみ使用

    explicit Texture3DDesc(I3D_COLOR_FORMAT _format=I3D_RGBA8, uvec3 _size=uvec3(0, 0, 0), uint32 _mipmap=0, void *_data=NULL, size_t _data_size=0)
        : format(_format)
        , size(_size)
        , mipmap(_mipmap)
        , data(_data)
        , data_size(_data_size)
    {}
};

struct istInterModule ShaderDesc
{
    const char *source;
    uint32 source_len;

    explicit ShaderDesc(const char *s=NULL, uint32 l=0) : source(s), source_len(l) {}
};
typedef ShaderDesc VertexShaderDesc;
typedef ShaderDesc GeometryShaderDesc;
typedef ShaderDesc PixelShaderDesc;

struct istInterModule ShaderProgramDesc
{
    VertexShader    *vs;
    PixelShader     *ps;
    GeometryShader  *gs;

    explicit ShaderProgramDesc(VertexShader *v=NULL, PixelShader *p=NULL, GeometryShader *g=NULL)
        : vs(v), ps(p), gs(g)
    {}
};

union ColorMask
{
    struct {
        uint8 red   : 1;
        uint8 green : 1;
        uint8 blue  : 1;
        uint8 alpha : 1;
    };
    uint8 mask;
};

struct istInterModule BlendStateDesc
{
    BlendStateDesc()
        : enable_blend(false)
        , equation_rgb(I3D_BLEND_ADD), equation_a(I3D_BLEND_ADD)
        , func_src_rgb(I3D_BLEND_ONE), func_src_a(I3D_BLEND_ONE), func_dst_rgb(I3D_BLEND_ZERO), func_dst_a(I3D_BLEND_ZERO)
    {
        for(int i=0; i<_countof(masks); ++i) { masks[i].mask=0xff; }
    }

    bool enable_blend;
    I3D_BLEND_EQUATION equation_rgb;
    I3D_BLEND_EQUATION equation_a;
    I3D_BLEND_FUNC func_src_rgb;
    I3D_BLEND_FUNC func_src_a;
    I3D_BLEND_FUNC func_dst_rgb;
    I3D_BLEND_FUNC func_dst_a;
    ColorMask masks[I3D_MAX_RENDER_TARGETS];
};

struct istInterModule DepthStencilStateDesc
{
    DepthStencilStateDesc()
        : depth_enable(false), depth_write(true), depth_func(I3D_DEPTH_LESS)
        , stencil_enable(false), stencil_func(I3D_STENCIL_ALWAYS), stencil_ref(0), stencil_mask(~0)
        , stencil_op_onsfail(I3D_STENCIL_KEEP), stencil_op_ondfail(I3D_STENCIL_KEEP), stencil_op_onpass(I3D_STENCIL_KEEP)
    {
    }

    bool                depth_enable;
    bool                depth_write;
    I3D_DEPTH_FUNC      depth_func;

    bool                stencil_enable;
    I3D_STENCIL_FUNC    stencil_func;
    int32               stencil_ref;
    uint32              stencil_mask;
    I3D_STENCIL_OP      stencil_op_onsfail;
    I3D_STENCIL_OP      stencil_op_ondfail;
    I3D_STENCIL_OP      stencil_op_onpass;
};


class istInterModule Viewport
{
public:
    Viewport() : m_pos(0,0), m_size(100,100) {}
    Viewport(const ivec2 pos, const uvec2 &size) : m_pos(pos), m_size(size) {}

    const ivec2& getPosition() const{ return m_pos; }
    const uvec2& getSize() const    { return m_size; }
    float32 getAspectRatio() const  { return (float32)m_size.x/(float32)m_size.y; }

    void setPosition(const ivec2 v) { m_pos=v; }
    void setSize(const ivec2 v)     { m_size=v; }

    bool operator==(const Viewport &v) const { return  memcmp(this, &v, sizeof(*this))==0; }

private:
    ivec2 m_pos;
    uvec2 m_size;
};


#define I3DGL_DECLARE_DEVICE_RESOURCE(classname) \
private:\
    istMakeDestructable;\
    friend class Device;\
    friend class DeviceContext;


} // namespace i3dgl
} // namespace ist

#endif // __ist_i3dgl_Types__
