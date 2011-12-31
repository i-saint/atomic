#ifndef __ist_DeviceResource__
#define __ist_DeviceResource__

namespace ist {
namespace i3d {


class DeviceResource
{
private:

    // non copyable
    DeviceResource(const DeviceResource&);
    DeviceResource& operator=(const DeviceResource&);

public:
    DeviceResource() {}
    virtual ~DeviceResource() {}
    virtual bool initialize() { return false; }
    virtual void finalize() {}
    virtual void OnDeviceLost() {}
    virtual void OnDeviceRestored() {}
};

} // namespace i3d
} // namespace ist
#endif // __ist_DeviceResource__
