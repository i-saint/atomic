#ifndef __ist_i3dgl_DeviceResource__
#define __ist_i3dgl_DeviceResource__

#include "i3dglTypes.h"

namespace ist {
namespace i3dgl {

class istInterModule Device;



class istInterModule DeviceResource : public SharedObject
{
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

#ifdef __i3d_enable_resource_leak_check__
public:
    int32 m_callstack_size;
    void *m_stack[16];
    void printLeakInfo();
#endif // __i3d_enable_leak_check__
};

} // namespace i3d
} // namespace ist
#endif // __ist_i3dgl_DeviceResource__
