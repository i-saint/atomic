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

private:
    GLuint m_buffer_object;
    GLuint m_size;

public:
    BufferObject();
    ~BufferObject();

    GLuint size() const;

    void bind() const;
    void unbind() const;

    void* lock(LOCK mode);
    void unlock();

    // data は NULL でもよく、その場合メモリ確保だけが行われる。
    void allocate(GLuint size, USAGE usage, void *data=NULL);
};

typedef BufferObject<GL_ARRAY_BUFFER>           VertexBufferObject;
typedef BufferObject<GL_ELEMENT_ARRAY_BUFFER>   IndexBufferObject;
typedef BufferObject<GL_PIXEL_PACK_BUFFER>      PixelPackBufferObject;
typedef BufferObject<GL_PIXEL_UNPACK_BUFFER>    PixelUnpackBufferObject;
typedef BufferObject<GL_UNIFORM_BUFFER>         UniformBufferObject;

} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_BufferObject__

