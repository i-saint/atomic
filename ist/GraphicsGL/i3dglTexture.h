#ifndef __ist_i3dgl_Texture__
#define __ist_i3dgl_Texture__

#include "i3dglTypes.h"
#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {

class Sampler : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Sampler)
typedef DeviceResource super;
private:
    SamplerDesc m_desc;

    Sampler(Device *dev, const SamplerDesc &desc);
    ~Sampler();

public:
    void bind(uint32 slot);
    void unbind(uint32 slot);
};


class Texture1D : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Texture1D)
typedef DeviceResource super;
private:
    static const int TEXTURE_TYPE;
    Texture1DDesc m_desc;

private:
    Texture1D(Device *dev, const Texture1DDesc &desc);
    ~Texture1D();

    void bind() const;
    void unbind() const;

public:
    void copy(uint32 mip_level, uint32 pos, uint32 size, I3D_COLOR_FORMAT format, void *data);
    void generateMipmap();

    void bind(int slot) const;
    void unbind(int slot) const;

    const Texture1DDesc& getDesc() const;
};


class Texture2D : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Texture2D)
typedef DeviceResource super;
private:
    static const int TEXTURE_TYPE;
    Texture2DDesc m_desc;

private:
    Texture2D(Device *dev, const Texture2DDesc &desc);
    ~Texture2D();

    void bind() const;
    void unbind() const;

public:
    void copy(uint32 mip_level, const uvec2 &pos, const uvec2 &size, I3D_COLOR_FORMAT format, void *data);
    void generateMipmap();

    void bind(int slot) const;
    void unbind(int slot) const;

    const Texture2DDesc& getDesc() const;
};


class Texture3D : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(Texture3D)
typedef DeviceResource super;
private:
    static const int TEXTURE_TYPE;
    Texture3DDesc m_desc;

private:
    Texture3D(Device *dev, const Texture3DDesc &desc);
    ~Texture3D();

    void bind() const;
    void unbind() const;

public:
    void copy(uint32 mip_level, const uvec3 &pos, const uvec3 &size, I3D_COLOR_FORMAT format, void *data);
    void generateMipmap();

    void bind(int slot) const;
    void unbind(int slot) const;

    const Texture3DDesc& getDesc() const;
};

} // namespace i3dgl
} // namespace ist

#endif // __ist_i3dgl_Texture__
