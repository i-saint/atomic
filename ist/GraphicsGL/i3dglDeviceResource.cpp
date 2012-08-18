#include "istPCH.h"
#ifdef __ist_with_OpenGL__
#include "ist/Base.h"
#include "ist/Debug.h"
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
    setRef(1);
    istSafeAddRef(m_device);
#ifdef __i3d_enable_resource_leak_check__
    m_callstack_size = GetCallstack(m_stack, _countof(m_stack));
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


#ifdef __i3d_enable_resource_leak_check__
void DeviceResource::printLeakInfo()
{
    stl::string text = CallstackToSymbolNames(m_stack, m_callstack_size);
    istPrint(text.c_str());
}
#endif // __i3d_enable_leak_check__

} // namespace i3d
} // namespace ist
#endif // __ist_with_OpenGL__
