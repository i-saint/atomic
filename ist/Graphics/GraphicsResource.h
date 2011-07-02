#ifndef __ist_GraphicsResource__
#define __ist_GraphicsResource__

namespace ist {
namespace graphics {


class GraphicsResource
{
private:
    // non copyable
    GraphicsResource(const GraphicsResource&);
    GraphicsResource& operator=(const GraphicsResource&);

public:
    GraphicsResource() {}
    virtual ~GraphicsResource() {}
    virtual bool initialize() { return false; }
    virtual void finalize() {}
    virtual void OnDeviceLost() {}
    virtual void OnDeviceRestored() {}
};

} // namespace graphics
} // namespace ist
#endif // __ist_GraphicsResource__
