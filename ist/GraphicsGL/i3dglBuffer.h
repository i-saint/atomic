#ifndef ist_i3dgl_Buffer_h
#define ist_i3dgl_Buffer_h

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {

class istInterModule Buffer : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Buffer);
typedef DeviceResource super;
friend class VertexArray;
public:
    const BufferDesc& getDesc() const { return m_desc; }

protected:
    Buffer(Device *dev, const BufferDesc &desc);
    ~Buffer();
    void bind() const;
    void unbind() const;
    void* map(I3D_MAP_MODE mode);
    void unmap();

    BufferDesc m_desc;
};


class istInterModule VertexArray : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(VertexArray);
typedef DeviceResource super;
public:
    static const int MAX_VERTEX_STREAM = 4;
    static const int MAX_VERTEX_DESC = 8;

    void setAttributes(uint32 vb_slot, Buffer *vbo, uint32 start, uint32 stride, const VertexDesc *descs, uint32 num_descs);

private:
    VertexArray(Device *dev);
    ~VertexArray();
    void bind() const;
    void unbind() const;
    bool isDirty() const { return m_dirty; }

private:
    struct VertexStreamDesc {
        VertexDesc vertex_descs[MAX_VERTEX_DESC];
        uint32 num_vertex_descs;
        uint32 start;
        uint32 stride;
        Buffer *buffer;
    };
    VertexStreamDesc m_stream_descs[MAX_VERTEX_STREAM];
    mutable bool m_dirty;
};

} // namespace i3d
} // namespace ist
#endif // ist_i3dgl_Buffer_h

