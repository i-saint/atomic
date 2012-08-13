#ifndef __ist_i3ddx11_RenderTarget__
#define __ist_i3ddx11_RenderTarget__

#include "i3ddx11Types.h"
#include "i3ddx11DeviceResource.h"

namespace ist {
namespace i3ddx11 {


class Sampler : public DeviceResource
{
I3DDX11_DECLARE_DEVICE_RESOURCE(Sampler)
private:
public:
};


class Texture2D : public DeviceResource
{
I3DDX11_DECLARE_DEVICE_RESOURCE(Texture2D)
private:
    ID3D11Texture2D *m_d3dtexture;
    ID3D11RenderTargetView *m_d3drtv;
    ID3D11DepthStencilView *m_d3ddsv;
    ID3D11ShaderResourceView *m_d3dsrv;
    uvec2 m_size;

private:
    Texture2D(Device *dev);
    ~Texture2D();

    ID3D11Texture2D*            getD3DTexture();
    ID3D11RenderTargetView*     getD3DRenderTargetView();
    ID3D11DepthStencilView*     getD3DDepthStencilView();
    ID3D11ShaderResourceView*   getD3DShaderResourceView();

public:
    bool allocate(const uvec2 &size, I3D_COLOR_FORMAT format, void *data=NULL);
    const uvec2& getSize() const;
};


} // namespace i3ddx11
} // namespace ist
#endif // __ist_i3ddx11_RenderTarget__
