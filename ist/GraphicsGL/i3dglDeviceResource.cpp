#include "istPCH.h"
#include "ist/Base.h"
#include "ist/Debug.h"
#ifdef ist_with_OpenGL
#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"
#include "i3dglDevice.h"

namespace ist {
namespace i3dgl {


DeviceResource::DeviceResource(Device *dev)
    : m_device(dev)
    , m_dr_handle(0)
    , m_handle(0)
{
    istSafeAddRef(m_device);
#ifdef i3d_enable_resource_leak_check
    m_callstack_size = GetCallstack(m_stack, _countof(m_stack), 3);
#endif // __i3d_enable_leak_check__
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

GLuint DeviceResource::getHandle() const
{
    return m_handle;
}


#ifdef i3d_enable_resource_leak_check
void DeviceResource::printLeakInfo()
{
    stl::string text = CallstackToSymbolNames(m_stack, m_callstack_size);
    istPrint(text.c_str());
}
#endif // __i3d_enable_leak_check__

} // namespace i3d
} // namespace ist
#endif // ist_with_OpenGL
