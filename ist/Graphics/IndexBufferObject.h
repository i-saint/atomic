#ifndef __ist_Graphics_IndexBufferObject__
#define __ist_Graphics_IndexBufferObject__

#include "GraphicsResource.h"

namespace ist {
namespace graphics {


class IndexBufferObject : public GraphicsResource
{
public:
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
    IndexBufferObject();
    ~IndexBufferObject();

    GLuint size() const;

    void bind() const;
    void unbind() const;

    void* lock(LOCK mode);
    void unlock();

    // data は NULL でもよく、その場合メモリ確保だけが行われる。
    void allocate(GLuint size, USAGE usage, void *data=NULL);
};


} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_IndexBufferObject__
