#ifndef __ist_DeviceResource__
#define __ist_DeviceResource__

#include "i3dTypes.h"

namespace ist {
namespace i3d {

class Device;

class DeviceResource
{
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

public:
    DeviceResource();
    virtual ~DeviceResource();

    int32 getRef() const;
    int32 addRef();
    int32 release();

    GLuint getHandle() const;
};

} // namespace i3d
} // namespace ist
#endif // __ist_DeviceResource__
