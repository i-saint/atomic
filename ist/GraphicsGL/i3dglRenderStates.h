#ifndef ist_i3dgl_RenderStates_h
#define ist_i3dgl_RenderStates_h

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {


class istAPI BlendState : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(BlendState);
typedef DeviceResource super;
public:
    bool operator==(const BlendState &v) const { return  istMemcmp(&m_desc, &v.m_desc, sizeof(m_desc))==0; }
    const BlendStateDesc& getDesc() const { return m_desc; }

private:
    BlendState(Device *dev, const BlendStateDesc &desc);

    void apply();
    BlendStateDesc m_desc;
};


class istAPI DepthStencilState : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(DepthStencilState);
typedef DeviceResource super;
public:
    bool operator==(const DepthStencilState &v) const { return  istMemcmp(&m_desc, &v.m_desc, sizeof(m_desc))==0; }
    const DepthStencilStateDesc& getDesc() const { return m_desc; }

private:
    DepthStencilState(Device *dev, const DepthStencilStateDesc &desc);

    void apply();
    DepthStencilStateDesc m_desc;
};


} // namespace i3dgl
} // namespace ist

#endif // ist_i3dgl_RenderStates_h
