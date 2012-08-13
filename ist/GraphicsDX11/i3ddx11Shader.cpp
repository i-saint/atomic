#include "stdafx.h"
#include "../Base.h"
#include <D3D11.h>
#include <D3DX11.h>
#include "i3ddx11Device.h"
#include "i3ddx11Shader.h"

#ifdef __ist_with_DirectX11__
namespace ist {
namespace i3ddx11 {


VertexShader::VertexShader( Device *dev )
    : DeviceResource(dev)
    , m_d3dshader(NULL)
{
}

VertexShader::~VertexShader()
{
    if(m_d3dshader) { m_d3dshader->Release(); }
}


PixelShader::PixelShader( Device *dev )
    : DeviceResource(dev)
    , m_d3dshader(NULL)
{
}

PixelShader::~PixelShader()
{
    if(m_d3dshader) { m_d3dshader->Release(); }
}


GeometryShader::GeometryShader( Device *dev )
    : DeviceResource(dev)
    , m_d3dshader(NULL)
{
}

GeometryShader::~GeometryShader()
{
    if(m_d3dshader) { m_d3dshader->Release(); }
}

} // namespace i3ddx11
} // namespace ist
#endif // __ist_with_DirectX11__
