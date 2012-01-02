#ifndef __ist_i3d_Buffer__
#define __ist_i3d_Buffer__

#include "i3dTypes.h"
#include "i3dDeviceResource.h"

namespace ist {
namespace i3d {

template<GLuint BufferType>
class Buffer : public DeviceResource
{
I3D_DECLARE_DEVICE_RESOURCE(Buffer);
protected:
    GLuint m_size;
    GLuint m_capacity;

    Buffer();
    ~Buffer();

public:
    // data は NULL でもよく、その場合メモリ確保だけが行われる。
    void allocate(GLuint size, I3D_USAGE usage, void *data=NULL);

    void bind() const;
    void unbind() const;

    void* map(I3D_MAP_MODE mode);
    void unmap();

    GLuint size() const;
};


class VertexBuffer : public Buffer<GL_ARRAY_BUFFER>
{
I3D_DECLARE_DEVICE_RESOURCE(VertexBuffer);
private:
    VertexBuffer() {}
    ~VertexBuffer() {}

public:
};

class IndexBuffer : public Buffer<GL_ELEMENT_ARRAY_BUFFER>
{
I3D_DECLARE_DEVICE_RESOURCE(IndexBuffer);
private:
    IndexBuffer() {}
    ~IndexBuffer() {}

public:
};

class PixelBuffer : public Buffer<GL_PIXEL_PACK_BUFFER>
{
I3D_DECLARE_DEVICE_RESOURCE(PixelBuffer);
private:
    PixelBuffer() {}
    ~PixelBuffer() {}

public:
};

class PixelUnpackBuffer : public Buffer<GL_PIXEL_UNPACK_BUFFER>
{
I3D_DECLARE_DEVICE_RESOURCE(PixelUnpackBuffer);
private:
    PixelUnpackBuffer() {}
    ~PixelUnpackBuffer() {}

public:
};

class UniformBuffer : public Buffer<GL_UNIFORM_BUFFER>
{
I3D_DECLARE_DEVICE_RESOURCE(UniformBuffer);
private:
    UniformBuffer() {}
    ~UniformBuffer() {}

public:
    void bindBase(GLuint index) const;
    void bindRange(GLuint index, GLintptr offset, GLsizeiptr size) const;
};


class VertexArray : public DeviceResource
{
I3D_DECLARE_DEVICE_RESOURCE(VertexArray);
private:
    VertexArray();
    ~VertexArray();

public:
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

