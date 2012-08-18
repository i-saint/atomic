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
private:
    Device *m_owner_device;
    ResourceHandle m_dr_handle;

protected:
    GLuint m_handle;

private:
    ResourceHandle getDeviceResourceHandle() const;
    void setDeviceResourceHandle(ResourceHandle v);

protected:
    DeviceResource(Device *dev);
    virtual ~DeviceResource();
    virtual void onZeroRef();
    Device* getOwnerDevice();

public:
    GLuint getHandle() const;
};

} // namespace i3d
} // namespace ist
#endif // __ist_i3dgl_DeviceResource__
