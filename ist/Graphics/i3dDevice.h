#ifndef __ist_i3d_Device__
#define __ist_i3d_Device__

#include "i3dBuffer.h"
#include "i3dDeviceResource.h"
#include "i3dRenderTarget.h"
#include "i3dShader.h"

namespace ist {
namespace i3d {

class Device
{
private:
#ifdef _WIN32
    HWND    m_hwnd;
    HDC     m_hdc;
    HGLRC   m_hglrc;
#endif // _WIN32

    std::vector<DeviceResource*>        m_resources;
    std::vector<ResourceHandle>   m_vacant;
    void addResource(DeviceResource *v);

public:
#ifdef _WIN32
    Device(HWND hwnd);
#endif // _WIN32
    ~Device();

    VertexBuffer*   createVertexBuffer();
    IndexBuffer*    createIndexBuffer();
    UniformBuffer*  createUniformBuffer();
    PixelBuffer*    createPixelBuffer();
    VertexArray*    createVertexArray();
    VertexShader*   createVertexShader();
    PixelShader*    createPixelShader();
    GeometryShader* createGeometryShader();
    void deleteResource(ResourceHandle v);

    void swapBuffers();
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3d_Device__
