#ifndef __ist_isd_Device__
#define __ist_isd_Device__

#include "isdTypes.h"

namespace ist {
namespace isd {

class istInterModule Device : public SharedObject
{
istMakeDestructable;
friend Device* CreateDevice();
private:
    ALCdevice *m_device;
    ALCcontext *m_context;
    stl::vector<DeviceResource*> m_resources;
    stl::vector<ResourceHandle> m_vacant;

    void addResource(DeviceResource *v);

    Device();
    ~Device();

public:
    Buffer* createBuffer();
    Source* createSource();
    Listener* createListener();

    void deleteResource(ResourceHandle v);
};
Device* CreateDevice();

} // namespace isd
} // namespace ist

#endif // __ist_isd_Device__
