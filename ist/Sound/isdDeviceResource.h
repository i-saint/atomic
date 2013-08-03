#ifndef __ist_isd_DeviceResource__
#define __ist_isd_DeviceResource__

#include "ist/Sound/isdTypes.h"

namespace ist {
namespace isd {

class Device;

class istAPI DeviceResource : public SharedObject
{
istMakeDestructable;
friend class Device;
public:
    virtual void onZeroRef();

    Device* getOwnerDevice();
    ResourceHandle getDeviceResourceHandle() const;
    void setDeviceResourceHandle(ResourceHandle v);

    ALuint getHandle() const;

protected:
    DeviceResource(Device *v);
    virtual ~DeviceResource();

    ALuint m_handle;

private:
    // non copyable
    DeviceResource(const DeviceResource&);
    DeviceResource& operator=(const DeviceResource&);

    Device *m_device;
    ResourceHandle m_dr_handle;
};

} // namespace isd
} // namespace ist

#endif // __ist_isd_DeviceResource__
