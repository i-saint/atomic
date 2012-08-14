#include "istPCH.h"
#ifdef __ist_with_DirectX11__
#include "../Base.h"
#include <D3D11.h>
#include <D3DX11.h>
#include "i3ddx11Device.h"
#include "i3ddx11DeviceResource.h"

namespace ist {
namespace i3ddx11 {


DeviceResource::DeviceResource(Device *dev)
    : m_owner_device(dev)
    , m_dr_handle(0)
{
    setRef(1);
}

DeviceResource::~DeviceResource()
{
    m_owner_device->deleteResource(getDeviceResourceHandle());
}

Device* DeviceResource::getOwnerDevice()
{
    return m_owner_device;
}

ResourceHandle DeviceResource::getDeviceResourceHandle() const
{
    return m_dr_handle;
}

void DeviceResource::setDeviceResourceHandle(ResourceHandle v)
{
    m_dr_handle = v;
}


} // namespace i3ddx11
} // namespace ist
#endif // __ist_with_DirectX11__
