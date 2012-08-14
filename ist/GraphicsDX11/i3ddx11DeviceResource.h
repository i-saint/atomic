#ifndef __ist_i3ddx11_DeviceResource__
#define __ist_i3ddx11_DeviceResource__

#include "ist/Base/SharedObject.h"
#include "ist/GraphicsDX11/i3ddx11Types.h"

namespace ist {
namespace i3ddx11 {

class Device;

class DeviceResource : public SharedObject
{
istMakeDestructable;
friend class Device;
public:
    ResourceHandle getDeviceResourceHandle() const;

protected:
    DeviceResource(Device *dev);
    virtual ~DeviceResource();

    Device* getOwnerDevice();

private:
    // non copyable
    DeviceResource(const DeviceResource&);
    DeviceResource& operator=(const DeviceResource&);

    void setDeviceResourceHandle(ResourceHandle v);

private:
    Device *m_owner_device;
    ResourceHandle m_dr_handle;
};

} // namespace i3ddx11
} // namespace ist
#endif // __ist_i3ddx11_DeviceResource__
