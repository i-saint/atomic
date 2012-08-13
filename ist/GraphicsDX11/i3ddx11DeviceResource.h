#ifndef __ist_i3ddx11_DeviceResource__
#define __ist_i3ddx11_DeviceResource__

#include "i3ddx11Types.h"

namespace ist {
namespace i3ddx11 {

class Device;

class DeviceResource
{
template<class T> friend T* ::call_destructor(T*);
friend class Device;
private:
    Device *m_owner_device;
    ResourceHandle m_dr_handle;
    int32 m_reference_count;

private:
    // non copyable
    DeviceResource(const DeviceResource&);
    DeviceResource& operator=(const DeviceResource&);

    void setDeviceResourceHandle(ResourceHandle v);
    ResourceHandle getDeviceResourceHandle() const;

protected:
    DeviceResource(Device *dev);
    virtual ~DeviceResource();

    Device* getOwnerDevice();

public:
    int32 getRef() const;
    int32 addRef();
    int32 release();
};

} // namespace i3ddx11
} // namespace ist
#endif // __ist_i3ddx11_DeviceResource__
