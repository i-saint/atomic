#include "stdafx.h"
#include "../Base.h"
#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"
#include "i3dglDevice.h"

namespace ist {
namespace i3dgl {


ReferenceCounter::ReferenceCounter()
    : m_reference_count(1)
{
}

ReferenceCounter::~ReferenceCounter()
{
}

void ReferenceCounter::onZeroRef()
{
    istDelete(this);
}

int32 ReferenceCounter::getRef() const
{
    return m_reference_count;
}

int32 ReferenceCounter::addRef()
{
    return ++m_reference_count;
}

int32 ReferenceCounter::release()
{
    if(--m_reference_count==0) {
        onZeroRef();
    }
    return m_reference_count;
}



DeviceResource::DeviceResource(Device *dev)
    : m_owner_device(dev)
    , m_dr_handle(0)
    , m_handle(0)
{
}

DeviceResource::~DeviceResource()
{
}

void DeviceResource::onZeroRef()
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

GLuint DeviceResource::getHandle() const
{
    return m_handle;
}

} // namespace i3d
} // namespace ist
