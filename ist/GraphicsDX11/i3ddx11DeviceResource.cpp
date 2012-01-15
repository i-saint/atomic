#include "stdafx.h"
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
    , m_reference_count(1)
{
}

DeviceResource::~DeviceResource()
{
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


int32 DeviceResource::getRef() const
{
    return m_reference_count;
}

int32 DeviceResource::addRef()
{
    return ++m_reference_count;
}

int32 DeviceResource::release()
{
    if(--m_reference_count==0) {
        m_owner_device->deleteResource(getDeviceResourceHandle());
    }
    return m_reference_count;
}

} // namespace i3ddx11
} // namespace ist
