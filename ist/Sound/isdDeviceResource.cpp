#include "istPCH.h"
#include "ist/Base.h"
#include "ist/Sound.h"

namespace ist {
namespace isd {

DeviceResource::DeviceResource()
    : m_owner_device(NULL)
    , m_dr_handle(0)
    , m_handle(0)
{
    setRef(1);
}

DeviceResource::~DeviceResource()
{
}

Device* DeviceResource::getOwnerDevice()
{
    return m_owner_device;
}

void DeviceResource::setOwnerDevice( Device *v )
{
    m_owner_device = v;
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
