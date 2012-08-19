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
#ifdef __ist_env_Windows__
    HWND    m_hwnd;
    HDC     m_hdc;
    HGLRC   m_hglrc;
#endif // __ist_env_Windows__
    stl::vector<DeviceResource*>    m_resources;
    stl::vector<ResourceHandle>     m_vacant;
    void addResource(DeviceResource *v);

#ifdef __ist_env_Windows__
    Device(HWND hwnd);
    friend Device* CreateDevice(HWND hwnd);
#else // __ist_env_Windows__
#endif // __ist_env_Windows__
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

#ifdef __ist_env_Windows__
    HDC getHDC() { return m_hdc; }
    HGLRC getHGLRC() { return m_hglrc; }
#endif // __ist_env_Windows__
#ifdef __i3d_enable_resource_leak_check__
    void printLeakInfo();
#endif // __i3d_enable_leak_check__
};

#ifdef __ist_env_Windows__
Device* CreateDevice(HWND hwnd);
#else // __ist_env_Windows__
#endif // __ist_env_Windows__

} // namespace i3dgl
} // namespace ist

#endif // __ist_i3dgl_Device__
