#ifndef __ist_isd_DeviceResource__
#define __ist_isd_DeviceResource__

#include "ist/Sound/isdTypes.h"

namespace ist {
namespace isd {

class Device;

class istInterModule DeviceResource : public SharedObject
{
istMakeDestructable;
friend class Device;
public:
    Device* getOwnerDevice();
    void setOwnerDevice(Device *v);
    ResourceHandle getDeviceResourceHandle() const;
    void setDeviceResourceHandle(ResourceHandle v);

    ALuint getHandle() const;

protected:
    DeviceResource();
    virtual ~DeviceResource();

    ALuint m_handle;

private:
    // non copyable
    DeviceResource(const DeviceResource&);
    DeviceResource& operator=(const DeviceResource&);

    Device *m_owner_device;
    ResourceHandle m_dr_handle;
};

} // namespace isd
} // namespace ist

#endif // __ist_isd_DeviceResource__
