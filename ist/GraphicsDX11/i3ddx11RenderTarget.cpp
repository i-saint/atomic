#include "stdafx.h"
#include "../Base.h"
#include <D3D11.h>
#include <D3DX11.h>
#include "i3ddx11Device.h"
#include "i3ddx11RenderTarget.h"

namespace ist {
namespace i3ddx11 {

Texture2D::Texture2D(Device *dev)
: DeviceResource(dev)
, m_d3dtexture(NULL)
, m_d3drtv(NULL)
, m_d3ddsv(NULL)
, m_d3dsrv(NULL)
{
}

Texture2D::~Texture2D()
{
    if(m_d3dsrv) { m_d3dsrv->Release(); }
    if(m_d3ddsv) { m_d3ddsv->Release(); }
    if(m_d3drtv) { m_d3drtv->Release(); }
    if(m_d3dtexture) { m_d3dtexture->Release(); }
}


ID3D11Texture2D* Texture2D::getD3DTexture()
{
    return m_d3dtexture;
}

ID3D11RenderTargetView* Texture2D::getD3DRenderTargetView()
{
    return m_d3drtv;
}

ID3D11DepthStencilView* Texture2D::getD3DDepthStencilView()
{
    return m_d3ddsv;
}

ID3D11ShaderResourceView* Texture2D::getD3DShaderResourceView()
{
    return m_d3dsrv;
}

bool Texture2D::allocate(const uvec2 &size, I3D_COLOR_FORMAT fmt, void *data)
{
    return true;
}

const uvec2& Texture2D::getSize() const { return m_size; }


} // namespace i3ddx11
} // namespace ist


