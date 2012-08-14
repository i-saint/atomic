#include "stdafx.h"
#ifdef __ist_with_DirectX11__
#include "ist/Base.h"
#include "ist/Window.h"
#include <D3D11.h>
#include <D3DX11.h>
#include "i3ddx11Device.h"

namespace ist {
namespace i3ddx11 {

Device::Device(HWND hwnd)
: m_hwnd(hwnd)
, m_d3device(NULL)
, m_d3dcontext(NULL)
, m_dxgiswapchain(NULL)
, m_rtview(NULL)
, m_error(I3D_ERROR_NONE)
{
    bool fullscreen = istGetAplication()->isFullscreen();
    uvec2 wsize = istGetAplication()->getWindowSize();

    D3D_DRIVER_TYPE         driver_type = D3D_DRIVER_TYPE_NULL;
    D3D_FEATURE_LEVEL       feature_level = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = S_OK;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_DRIVER_TYPE driverTypes[] =
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    UINT numDriverTypes = ARRAYSIZE( driverTypes );

    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
    };
    UINT numFeatureLevels = ARRAYSIZE( featureLevels );

    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof( sd ) );
    sd.BufferCount = 1;
    sd.BufferDesc.Width = wsize.x;
    sd.BufferDesc.Height = wsize.y;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = !fullscreen;

    for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
    {
        driver_type = driverTypes[driverTypeIndex];
        hr = D3D11CreateDeviceAndSwapChain( NULL, driver_type, NULL, createDeviceFlags, featureLevels, numFeatureLevels,
            D3D11_SDK_VERSION, &sd, &m_dxgiswapchain, &m_d3device, &feature_level, &m_d3dcontext );
        if( SUCCEEDED( hr ) ) {
            break;
        }
    }
    if( FAILED( hr ) ) {
        m_error = I3D_ERROR_D3D11CreateDeviceAndSwapChain_Failed;
        return;
    }

    // Create a render target view
    ID3D11Texture2D* pBackBuffer = NULL;
    hr = m_dxgiswapchain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), ( LPVOID* )&pBackBuffer );
    if( FAILED( hr ) ) {
        m_error = I3D_ERROR_GetBuffer_Failed;
        return;
    }

    hr = m_d3device->CreateRenderTargetView( pBackBuffer, NULL, &m_rtview );
    pBackBuffer->Release();
    if( FAILED( hr ) ) {
        m_error = I3D_ERROR_CreateRenderTargetView_Failed;
        return;
    }

    m_d3dcontext->OMSetRenderTargets( 1, &m_rtview, NULL );

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = (FLOAT)wsize.x;
    vp.Height = (FLOAT)wsize.y;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    m_d3dcontext->RSSetViewports( 1, &vp );
}

Device::~Device()
{
    for(uint32 i=0; i<m_resources.size(); ++i) {
        istSafeDelete(m_resources[i]);
    }

    if( m_d3dcontext )      { m_d3dcontext->ClearState(); }
    if( m_rtview )          { m_rtview->Release(); }
    if( m_dxgiswapchain )   { m_dxgiswapchain->Release(); }
    if( m_d3dcontext )      { m_d3dcontext->Release(); }
    if( m_d3device )        { m_d3device->Release(); }
}


void Device::addResource( DeviceResource *v )
{
    if(!v) { return; }

    if(!m_vacant.empty()) {
        ResourceHandle drh = m_vacant.back();
        m_vacant.pop_back();
        m_resources[drh] = v;
        v->setDeviceResourceHandle(drh);
    }
    else {
        v->setDeviceResourceHandle(m_resources.size());
        m_resources.push_back(v);
    }
}

void Device::deleteResource( ResourceHandle v )
{
    istSafeDelete(m_resources[v]);
    m_vacant.push_back(v);
}

Buffer* Device::createBuffer()
{
    Buffer *r = istNew(Buffer)(this);
    addResource(r);
    return r;
}

VertexShader* Device::createVertexShader()
{
    VertexShader *r = istNew(VertexShader)(this);
    addResource(r);
    return r;
}

PixelShader* Device::createPixelShader()
{
    PixelShader *r = istNew(PixelShader)(this);
    addResource(r);
    return r;
}

GeometryShader* Device::createGeometryShader()
{
    GeometryShader *r = istNew(GeometryShader)(this);
    addResource(r);
    return r;
}

Texture2D* Device::createTexture2D()
{
    Texture2D *r = istNew(Texture2D)(this);
    addResource(r);
    return r;
}

void Device::swapBuffers()
{
    m_dxgiswapchain->Present( 0, 0 );
}

} // namespace i3ddx11
} // namespace ist
#endif // __ist_with_DirectX11__
