#include "istPCH.h"
#include "ist/Base.h"
#ifdef ist_with_OpenGL
#include "i3dglBuffer.h"

namespace ist {
namespace i3dgl {

Buffer::Buffer(Device *dev, const BufferDesc &desc)
: super(dev)
, m_desc(desc)
{
    glGenBuffers(1, &m_handle);

    glBindBuffer(m_desc.type, m_handle);
    glBufferData(m_desc.type, m_desc.size, m_desc.data, m_desc.usage);
    glBindBuffer(m_desc.type, 0);
}

Buffer::~Buffer()
{
    if(m_handle!=0) {
        glDeleteBuffers(1, &m_handle);
        m_handle = 0;
    }
}

void Buffer::bind() const
{
    glBindBuffer(m_desc.type, m_handle);
}

void Buffer::unbind() const
{
    glBindBuffer(m_desc.type, 0);
}

void* Buffer::map(I3D_MAP_MODE mode)
{
    glBindBuffer(m_desc.type, m_handle);
    void *r = glMapBuffer(m_desc.type, mode);
    istAssert(r!=NULL);
    glBindBuffer(m_desc.type, 0);
    return r;
}

void Buffer::unmap()
{
    glBindBuffer(m_desc.type, m_handle);
    glUnmapBuffer(m_desc.type);
    glBindBuffer(m_desc.type, 0);
}



VertexArray::VertexArray(Device *dev)
    : super(dev)
    , m_dirty(false)
{
    istMemset(m_stream_descs, 0, sizeof(m_stream_descs));
    glGenVertexArrays(1, &m_handle);
}

VertexArray::~VertexArray()
{
    if(m_handle!=0) {
        glDeleteVertexArrays(1, &m_handle);
        m_handle = 0;
    }
}

void VertexArray::bind() const
{
    m_dirty = false;
    glBindVertexArray(m_handle);
    for(size_t si=0; si<MAX_VERTEX_STREAM; ++si) {
        const VertexStreamDesc &vsd = m_stream_descs[si];
        if(!vsd.buffer) { continue; }
        vsd.buffer->bind();
        for(size_t vi=0; vi<vsd.num_vertex_descs; ++vi) {
            const VertexDesc& desc = vsd.vertex_descs[vi];
            glEnableVertexAttribArray(desc.location);
            // float type
            if(desc.type==I3D_HALF || desc.type==I3D_FLOAT || desc.type==I3D_DOUBLE || desc.normalize) {
                glVertexAttribPointer(desc.location, desc.num_elements, desc.type, desc.normalize, vsd.stride, (GLvoid*)desc.offset);
            }
            // integer type
            else {
                glVertexAttribIPointer(desc.location, desc.num_elements, desc.type, vsd.stride, (GLvoid*)desc.offset);
            }
            glVertexAttribDivisor(desc.location, desc.divisor);
        }
    }
}

void VertexArray::unbind() const
{
    glBindVertexArray(0);
}

void VertexArray::setAttributes( uint32 vb_slot, Buffer* vb, uint32 stride, const VertexDesc *descs, uint32 num_descs )
{
    istAssert(vb_slot<MAX_VERTEX_STREAM);
    istAssert(num_descs<MAX_VERTEX_DESC);
    m_dirty = true;
    VertexStreamDesc &vsd = m_stream_descs[vb_slot];
    vsd.stride = stride;
    vsd.num_vertex_descs = num_descs;
    vsd.buffer = vb;
    std::copy(descs, descs+num_descs, vsd.vertex_descs);
}


} // namespace i3d
} // namespace ist
#endif // ist_with_OpenGL
