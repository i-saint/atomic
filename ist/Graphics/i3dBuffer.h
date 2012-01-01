#ifndef __ist_i3d_Buffer__
#define __ist_i3d_Buffer__

#include "i3dTypes.h"
#include "i3dDeviceResource.h"

namespace ist {
namespace i3d {

template<GLuint BufferType>
class Buffer : public DeviceResource
{
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
    void allocate(GLuint size, I3D_USAGE usage, void *data=NULL);

    void bind() const;
    void unbind() const;

    void* map(I3D_MAP_MODE mode);
    void unmap();

    GLuint size() const;
    GLuint getHandle() const { return m_handle; }
};

typedef Buffer<GL_ARRAY_BUFFER>           VertexBuffer;
typedef Buffer<GL_ELEMENT_ARRAY_BUFFER>   IndexBuffer;
typedef Buffer<GL_PIXEL_PACK_BUFFER>      PixelBuffer;
typedef Buffer<GL_PIXEL_UNPACK_BUFFER>    PixelUnpackBuffer;

class UniformBuffer : public Buffer<GL_UNIFORM_BUFFER>
{
public:
    void bindBase(GLuint index) const;
    void bindRange(GLuint index, GLintptr offset, GLsizeiptr size) const;
};


class VertexArray : public DeviceResource
{
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
    void setAttribute(GLuint index, GLint num_elements, VertexBuffer &vbo);
    void setInstanceAttribute(GLuint i, GLint num_elements, VertexBuffer &vbo);

    void setAttributes(VertexBuffer& vbo, size_t stride, const VertexDescriptor *descs, size_t num_descs);
};

} // namespace i3d
} // namespace ist
#endif // __ist_i3d_Buffer__

