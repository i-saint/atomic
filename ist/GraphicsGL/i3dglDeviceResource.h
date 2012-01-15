#ifndef __ist_i3dgl_DeviceResource__
#define __ist_i3dgl_DeviceResource__

#include "i3dglTypes.h"

namespace ist {
namespace i3dgl {

class Device;

class ReferenceCounter
{
private:
    int32 m_reference_count;

    // non copyable
    ReferenceCounter(const ReferenceCounter&);
    ReferenceCounter& operator=(const ReferenceCounter&);

protected:
    virtual void onZeroRef();

public:
    ReferenceCounter();
    virtual ~ReferenceCounter();
    int32 getRef() const;
    int32 addRef();
    int32 release();
};


class DeviceResource : public ReferenceCounter
{
template<class T> friend T* ::call_destructor(T*);
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
