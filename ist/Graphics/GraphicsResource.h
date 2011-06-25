#ifndef __ist_GraphicsResource__
#define __ist_GraphicsResource__

namespace ist {
namespace graphics {


class GraphicsResource
{
public:
    virtual ~GraphicsResource() {}
    virtual void OnDeviceLost() {}
    virtual void OnDeviceRestored() {}
};

} // namespace graphics
} // namespace ist
#endif // __ist_GraphicsResource__
