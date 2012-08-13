#ifndef __ist_i3ddx11_Shader__
#define __ist_i3ddx11_Shader__

#include "i3ddx11DeviceResource.h"

namespace ist {
namespace i3ddx11 {

class VertexShader : public DeviceResource
{
I3DDX11_DECLARE_DEVICE_RESOURCE(VertexShader);
private:
    ID3D11VertexShader *m_d3dshader;

    VertexShader(Device *dev);
    ~VertexShader();

public:
};

class PixelShader : public DeviceResource
{
I3DDX11_DECLARE_DEVICE_RESOURCE(PixelShader);
private:
    ID3D11PixelShader *m_d3dshader;

    PixelShader(Device *dev);
    ~PixelShader();

public:
};

class GeometryShader : public DeviceResource
{
I3DDX11_DECLARE_DEVICE_RESOURCE(GeometryShader);
private:
    ID3D11GeometryShader *m_d3dshader;

    GeometryShader(Device *dev);
    ~GeometryShader();

public:
};



} // namespace i3ddx11
} // namespace ist
#endif // __ist_i3ddx11_Shader__
