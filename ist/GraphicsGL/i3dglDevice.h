#ifndef __ist_i3dgl_Device__
#define __ist_i3dgl_Device__

#include "i3dglBuffer.h"
#include "i3dglDeviceResource.h"
#include "i3dglTexture.h"
#include "i3dglRenderTarget.h"
#include "i3dglShader.h"

namespace ist {
namespace i3dgl {

class istInterModule Device : public SharedObject
{
istMakeDestructable;
private:
#ifdef ist_env_Windows
    HWND    m_hwnd;
    HDC     m_hdc;
    HGLRC   m_hglrc;
#endif // ist_env_Windows
    stl::vector<DeviceResource*>    m_resources;
    stl::vector<ResourceHandle>     m_vacant;
    void addResource(DeviceResource *v);

#ifdef ist_env_Windows
    Device(HWND hwnd);
    friend Device* CreateDevice(HWND hwnd);
#else // ist_env_Windows
#endif // ist_env_Windows
    ~Device();
public:
    DeviceContext*  createContext();

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

#ifdef ist_env_Windows
    HDC getHDC() { return m_hdc; }
    HGLRC getHGLRC() { return m_hglrc; }
#endif // ist_env_Windows
#ifdef i3d_enable_resource_leak_check
    void printLeakInfo();
#endif // __i3d_enable_leak_check__
};

#ifdef ist_env_Windows
Device* CreateDevice(HWND hwnd);
#else // ist_env_Windows
#endif // ist_env_Windows

} // namespace i3dgl
} // namespace ist

#endif // __ist_i3dgl_Device__
