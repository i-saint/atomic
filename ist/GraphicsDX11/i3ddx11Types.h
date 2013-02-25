#ifndef __ist_i3ddx11_Types__
#define __ist_i3ddx11_Types__

#include "ist/Base.h"

namespace ist {
namespace i3ddx11 {

enum I3D_ERROR_CODE {
    I3D_ERROR_NONE,
    I3D_ERROR_D3D11CreateDeviceAndSwapChain_Failed,
    I3D_ERROR_GetBuffer_Failed,
    I3D_ERROR_CreateRenderTargetView_Failed,
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
    I3D_R8U,
    I3D_R16F,
    I3D_R32F,
    I3D_RG8U,
    I3D_RG16F,
    I3D_RG32F,
    I3D_RGB8U,  // texture only
    I3D_RGB16F, // 
    I3D_RGB32F, // 
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
    I3D_INT8    = GL_BYTE,
    I3D_UINT8   = GL_UNSIGNED_BYTE,
    I3D_INT16   = GL_SHORT,
    I3D_UINT16  = GL_UNSIGNED_SHORT,
    I3D_INT32   = GL_INT,
    I3D_UINT32  = GL_UNSIGNED_INT,
    I3D_FLOAT32 = GL_FLOAT,
    I3D_FLOAT64 = GL_DOUBLE,
};

enum I3D_CONSTANTS
{
    I3D_MAX_RENDER_TARGETS = 8,
};

struct VertexDescriptor
{
    GLuint location;
    I3D_TYPE type;
    GLuint num_elements; // must be 1,2,3,4
    GLuint offset;
    bool normalize;
    GLuint divisor; // 0: per vertex, other: per n instance
};

typedef uint32 ResourceHandle;
class Device;
class DeviceContext;
class DeviceResource;
class VertexBuffer;
class IndexBuffer;
class PixelBuffer;
class PixelUnpackBuffer;
class UniformBuffer;
class VertexArray;
class Texture1D;
class Texture2D;
class Texture3D;
class RenderBuffer;
class RenderTarget;
class VertexShader;
class PixelShader;
class GeometryShader;
class ShaderProgtam;

} // namespace i3ddx11
} // namespace ist

#define I3DDX11_DECLARE_DEVICE_RESOURCE(classname) \
private:\
    istMakeDestructable;\
    friend class Device;\
    friend class DeviceContext;

#endif // __ist_i3ddx11_Types__
