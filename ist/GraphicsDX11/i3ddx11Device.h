#ifndef __ist_i3ddx11_Device__
#define __ist_i3ddx11_Device__

#include "i3ddx11Buffer.h"
#include "i3ddx11DeviceResource.h"
#include "i3ddx11RenderTarget.h"
#include "i3ddx11Shader.h"

namespace ist {
namespace i3ddx11 {

class Device
{
private:
    HWND                    m_hwnd;
    ID3D11Device            *m_d3device;
    ID3D11DeviceContext     *m_d3dcontext;
    IDXGISwapChain          *m_dxgiswapchain;
    ID3D11RenderTargetView  *m_rtview;
    I3D_ERROR_CODE m_error;

    std::vector<DeviceResource*>        m_resources;
    std::vector<ResourceHandle>   m_vacant;
    void addResource(DeviceResource *v);

public:
    Device(HWND hwnd);
    ~Device();

    Buffer*         createBuffer();
    Texture2D*      createTexture2D();

    VertexShader*   createVertexShader();
    PixelShader*    createPixelShader();
    GeometryShader* createGeometryShader();

    void deleteResource(ResourceHandle v);

    void swapBuffers();

public:
    ID3D11Device* getD3DDevice() { return m_d3device; }
    ID3D11DeviceContext* getD3DContext() { return m_d3dcontext; }
};

} // namespace i3ddx11
} // namespace ist

#endif // __ist_i3ddx11_Device__
