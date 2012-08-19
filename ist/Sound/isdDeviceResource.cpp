#include "istPCH.h"
#include "ist/Base.h"
#include "ist/Sound.h"

namespace ist {
namespace isd {

DeviceResource::DeviceResource(Device *v)
    : m_device(v)
    , m_dr_handle(0)
    , m_handle(0)
{
    setRef(1);
    istSafeAddRef(m_device);
}

DeviceResource::~DeviceResource()
{
    istSafeRelease(m_device);
}

void DeviceResource::onZeroRef()
{
    m_device->deleteResource(getDeviceResourceHandle());
}

Device* DeviceResource::getOwnerDevice()
{
    return m_device;
}

ResourceHandle DeviceResource::getDeviceResourceHandle() const
{
    return m_dr_handle;
}

void DeviceResource::setDeviceResourceHandle(ResourceHandle v)
{
    m_dr_handle = v;
}

ALuint DeviceResource::getHandle() const
{
    return m_handle;
}

} // namespace isd
} // namespace ist
