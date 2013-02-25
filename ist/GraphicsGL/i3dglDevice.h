#ifndef ist_i3dgl_Device_h
#define ist_i3dgl_Device_h

#include "i3dglBuffer.h"
#include "i3dglDeviceResource.h"
#include "i3dglTexture.h"
#include "i3dglRenderTarget.h"
#include "i3dglShader.h"
#include "i3dglRenderStates.h"

namespace ist {
namespace i3dgl {

class istInterModule Device : public SharedObject
{
istNonCopyable(Device);
istMakeDestructable;
public:
    DeviceContext*  createImmediateContext();

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

    BlendState*         createBlendState(const BlendStateDesc &desc);
    DepthStencilState*  createDepthStencilState(const DepthStencilStateDesc &desc);

    void deleteResource(ResourceHandle v);
    void swapBuffers();

    DeviceContext* getImmediateContext();

#ifdef i3d_enable_resource_leak_check
    void printLeakInfo();
#endif // i3d_enable_leak_check

private:
    DeviceContext                   *m_immediate_context;
    stl::vector<DeviceResource*>    m_resources;
    stl::vector<ResourceHandle>     m_vacant;
    void addResource(DeviceResource *v);


#ifdef ist_env_Windows
public:
    HDC getHDC() { return m_hdc; }
    HGLRC getHGLRC() { return m_hglrc; }

private:
    friend Device* CreateDevice(HWND hwnd);
    Device(HWND hwnd);
    ~Device();

    HWND    m_hwnd;
    HDC     m_hdc;
    HGLRC   m_hglrc;
#endif // ist_env_Windows
};

Device* GetDevice();

#ifdef ist_env_Windows
Device* CreateDevice(HWND hwnd);
#else // ist_env_Windows
#endif // ist_env_Windows

} // namespace i3dgl
} // namespace ist

#endif // ist_i3dgl_Device_h
