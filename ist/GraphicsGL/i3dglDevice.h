#ifndef __ist_i3dgl_Device__
#define __ist_i3dgl_Device__

#include "i3dglBuffer.h"
#include "i3dglDeviceResource.h"
#include "i3dglTexture.h"
#include "i3dglRenderTarget.h"
#include "i3dglShader.h"

namespace ist {
namespace i3dgl {

class Device
{
private:
#ifdef istWindows
    HWND    m_hwnd;
    HDC     m_hdc;
    HGLRC   m_hglrc;
#endif // istWindows
    DeviceContext *m_context;

    stl::vector<DeviceResource*>    m_resources;
    stl::vector<ResourceHandle>     m_vacant;
    void addResource(DeviceResource *v);

public:
#ifdef istWindows
    Device(HWND hwnd);
#endif // istWindows
    ~Device();
    DeviceContext* getContext() { return m_context; }

    Buffer*         createBuffer(const BufferDesc &desc);
    VertexArray*    createVertexArray();

    VertexShader*   createVertexShader(const VertexShaderDesc &desc);
    PixelShader*    createPixelShader(const PixelShaderDesc &desc);
    GeometryShader* createGeometryShader(const GeometryShaderDesc &desc);
    ShaderProgram*  createShaderProgram(const ShaderProgramDesc &desc);

    Sampler*        createSampler(const SamplerDesc &desc);
    Texture1D*      createTexture1D(const Texture1DDesc &desc);
    Texture2D*      createTexture2D(const Texture2DDesc &desc);
    Texture3D*      createTexture3D(const Texture3DDesc &desc);
    RenderTarget*   createRenderTarget();

    void deleteResource(ResourceHandle v);

    void swapBuffers();

#ifdef istWindows
    HDC getHDC() { return m_hdc; }
    HGLRC getHGLRC() { return m_hglrc; }
#endif // istWindows
};

} // namespace i3d
} // namespace ist

#endif // __ist_i3dgl_Device__
