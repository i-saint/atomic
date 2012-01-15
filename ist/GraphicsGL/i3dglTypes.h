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
    I3D_BYTE    = GL_BYTE,
    I3D_UBYTE   = GL_UNSIGNED_BYTE,
    I3D_SHORT   = GL_SHORT,
    I3D_USHORT  = GL_UNSIGNED_SHORT,
    I3D_INT     = GL_INT,
    I3D_UINT    = GL_UNSIGNED_INT,
    I3D_FLOAT   = GL_FLOAT,
    I3D_DOUBLE  = GL_DOUBLE,
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

} // namespace i3d
} // namespace ist

#define I3DGL_DECLARE_DEVICE_RESOURCE(classname) \
private:\
    template<class T> friend T* ::call_destructor(T *v);\
    friend class Device;\
    friend class DeviceContext;

#endif // __ist_i3dgl_Types__
