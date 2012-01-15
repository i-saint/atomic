#include "stdafx.h"
#include "../Base.h"
#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"
#include "i3dglDevice.h"

namespace ist {
namespace i3dgl {


DeviceResource::DeviceResource()
    : m_owner_device(NULL)
    , m_dr_handle(0)
    , m_reference_count(1)
    , m_handle(0)
{
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

GLuint DeviceResource::getHandle() const
{
    return m_handle;
}

} // namespace i3d
} // namespace ist
