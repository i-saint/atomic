#ifndef __ist_i3dgl_Device__
#define __ist_i3dgl_Device__

#include "i3dglBuffer.h"
#include "i3dglDeviceResource.h"
#include "i3dglRenderTarget.h"
#include "i3dglShader.h"

namespace ist {
namespace i3dgl {

class Device
{
private:
#ifdef _WIN32
    HWND    m_hwnd;
    HDC     m_hdc;
    HGLRC   m_hglrc;
#endif // _WIN32
    DeviceContext *m_context;

    std::vector<DeviceResource*>        m_resources;
    std::vector<ResourceHandle>   m_vacant;
    void addResource(DeviceResource *v);

public:
#ifdef _WIN32
    Device(HWND hwnd);
#endif // _WIN32
    ~Device();
    DeviceContext* getContext() { return m_context; }

    Buffer*         createBuffer(const BufferDesc &desc);
    VertexArray*    createVertexArray();

    VertexShader*   createVertexShader();
    PixelShader*    createPixelShader();
    GeometryShader* createGeometryShader();
    ShaderProgram*  createShaderProgram();

    Texture1D*      createTexture1D();
    Texture2D*      createTexture2D();
    RenderTarget*   createRenderTarget();

    void deleteResource(ResourceHandle v);

    void swapBuffers();

#ifdef _WIN32
    HDC getHDC() { return m_hdc; }
    HGLRC getHGLRC() { return m_hglrc; }
#endif // _WIN32
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_Device__
