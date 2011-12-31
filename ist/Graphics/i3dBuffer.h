#ifndef __ist_i3d_Buffer__
#define __ist_i3d_Buffer__

#include "i3dResource.h"

namespace ist {
namespace i3d {

template<GLuint BufferType>
class Buffer : public DeviceResource
{
public:
    enum {
        BUFFER_TYPE = BufferType,
    };
    enum USAGE {
        USAGE_STATIC  = GL_STATIC_DRAW,
        USAGE_DYNAMIC = GL_DYNAMIC_DRAW,
        USAGE_STREAM  = GL_STREAM_DRAW,
    };
    enum MAP_MODE {
        MAP_READ       = GL_READ_ONLY,
        MAP_WRITE      = GL_WRITE_ONLY,
        MAP_READWRITE  = GL_READ_WRITE,
    };

protected:
    GLuint m_handle;
    GLuint m_size;
    GLuint m_capacity;

public:
    Buffer();
    ~Buffer();

    bool initialize();
    void finalize();

    // data は NULL でもよく、その場合メモリ確保だけが行われる。
    void allocate(GLuint size, USAGE usage, void *data=NULL);

    void bind() const;
    void unbind() const;

    void* map(MAP_MODE mode);
    void unmap();

    GLuint size() const;
    GLuint getHandle() const { return m_handle; }
};

typedef Buffer<GL_ARRAY_BUFFER>           VertexBufferObject;
typedef Buffer<GL_ELEMENT_ARRAY_BUFFER>   IndexBufferObject;
typedef Buffer<GL_PIXEL_PACK_BUFFER>      PixelPackBufferObject;
typedef Buffer<GL_PIXEL_UNPACK_BUFFER>    PixelUnpackBufferObject;

class UniformBufferObject : public Buffer<GL_UNIFORM_BUFFER>
{
public:
    void bindBase(GLuint index) const;
    void bindRange(GLuint index, GLintptr offset, GLsizeiptr size) const;
};


class VertexArray : public DeviceResource
{
public:
    enum TYPE {
        TYPE_BYTE   = GL_BYTE,
        TYPE_UBYTE  = GL_UNSIGNED_BYTE,
        TYPE_SHORT  = GL_SHORT,
        TYPE_USHORT = GL_UNSIGNED_SHORT,
        TYPE_INT    = GL_INT,
        TYPE_UINT   = GL_UNSIGNED_INT,
        TYPE_FLOAT  = GL_FLOAT,
        TYPE_DOUBLE = GL_DOUBLE,
    };
    struct Descriptor
    {
        GLuint location;
        TYPE type;
        GLuint num_elements; // must be 1,2,3,4
        GLuint offset;
        bool normalize;
        GLuint divisor; // 0: per vertex, other: per n instance
    };

private:
    GLuint m_handle;

public:
    VertexArray();
    ~VertexArray();
    bool initialize();
    void finalize();

    void bind() const;
    void unbind() const;

    // num_elements: 1,2,3,4
    void setAttribute(GLuint index, GLint num_elements, VertexBufferObject &vbo);
    void setInstanceAttribute(GLuint i, GLint num_elements, VertexBufferObject &vbo);

    void setAttributes(VertexBufferObject& vbo, size_t stride, const Descriptor *descs, size_t num_descs);
};

} // namespace i3d
} // namespace ist
#endif // __ist_i3d_Buffer__

