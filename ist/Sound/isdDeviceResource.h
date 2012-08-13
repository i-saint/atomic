#ifndef __ist_isd_DeviceResource__
#define __ist_isd_DeviceResource__

#include "isdTypes.h"

namespace ist {
namespace isd {

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
    ALuint m_handle;

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

    ALuint getHandle() const;
};

} // namespace isd
} // namespace ist

#endif // __ist_isd_DeviceResource__
