#ifndef __ist_i3dgl_DeviceResource__
#define __ist_i3dgl_DeviceResource__

#include "i3dglTypes.h"

namespace ist {
namespace i3dgl {

class Device;

class DeviceResource
{
template<class T> friend T* ::call_destructor(T*);
friend class Device;
private:
    Device *m_owner_device;
    ResourceHandle m_dr_handle;
    int32 m_reference_count;

protected:
    GLuint m_handle;

private:
    // non copyable
    DeviceResource(const DeviceResource&);
    DeviceResource& operator=(const DeviceResource&);

    Device* getOwnerDevice();
    void setOwnerDevice(Device *v);
    ResourceHandle getDeviceResourceHandle() const;
    void setDeviceResourceHandle(ResourceHandle v);

protected:
    DeviceResource();
    virtual ~DeviceResource();

public:
    int32 getRef() const;
    int32 addRef();
    int32 release();

    GLuint getHandle() const;
};

} // namespace i3d
} // namespace ist
#endif // __ist_i3dgl_DeviceResource__
