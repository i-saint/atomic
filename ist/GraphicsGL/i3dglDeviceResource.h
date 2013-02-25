#ifndef ist_i3dgl_DeviceResource_h
#define ist_i3dgl_DeviceResource_h

#include "i3dglTypes.h"

namespace ist {
namespace i3dgl {

class istInterModule Device;



class istInterModule DeviceResource : public SharedObject
{
istNonCopyable(DeviceResource);
istMakeDestructable;
friend class Device;
public:
    GLuint getHandle() const;

protected:
    DeviceResource(Device *dev);
    virtual ~DeviceResource();
    virtual void onZeroRef();
    Device* getOwnerDevice();

    GLuint m_handle;

private:
    Device *m_device;
    ResourceHandle m_dr_handle;

    // non-copyable
    ResourceHandle getDeviceResourceHandle() const;
    void setDeviceResourceHandle(ResourceHandle v);

#ifdef i3d_enable_resource_leak_check
public:
    int32 m_callstack_size;
    void *m_stack[ist_leak_check_max_callstack_size];
    void printLeakInfo();
#endif // __i3d_enable_leak_check__
};

} // namespace i3d
} // namespace ist
#endif // ist_i3dgl_DeviceResource_h
