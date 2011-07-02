#ifndef __ist_Graphics_BufferObject__
#define __ist_Graphics_BufferObject__

#include "GraphicsResource.h"

namespace ist {
namespace graphics {

template<GLuint BufferType>
class BufferObject : public GraphicsResource
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
    enum LOCK {
        LOCK_READ       = GL_READ_ONLY,
        LOCK_WRITE      = GL_WRITE_ONLY,
        LOCK_READWRITE  = GL_READ_WRITE,
    };

protected:
    GLuint m_handle;
    GLuint m_size;

public:
    BufferObject();
    ~BufferObject();

    bool initialize();
    void finalize();

    void bind() const;
    void unbind() const;

    void* lock(LOCK mode);
    void unlock();

    // data は NULL でもよく、その場合メモリ確保だけが行われる。
    void allocate(GLuint size, USAGE usage, void *data=NULL);

    GLuint size() const;
    GLuint getHandle() const { return m_handle; }
};

typedef BufferObject<GL_ARRAY_BUFFER>           VertexBufferObject;
typedef BufferObject<GL_ELEMENT_ARRAY_BUFFER>   IndexBufferObject;
typedef BufferObject<GL_PIXEL_PACK_BUFFER>      PixelPackBufferObject;
typedef BufferObject<GL_PIXEL_UNPACK_BUFFER>    PixelUnpackBufferObject;

class UniformBufferObject : public BufferObject<GL_UNIFORM_BUFFER>
{
public:
    void bindBase(GLuint index) const;
    void bindRange(GLuint index, GLintptr offset, GLsizeiptr size) const;
};

} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_BufferObject__

