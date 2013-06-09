#include "istPCH.h"
#include "ist/Base.h"
#ifdef ist_with_OpenGL
#include "i3dglDevice.h"
#include "i3dglDeviceContext.h"

namespace ist {
namespace i3dgl {

static Device *g_the_device = NULL;

#ifdef ist_env_Windows

struct Device::Members
{
    DeviceContext                  *immediate_context;
    stl::vector<DeviceResource*>    resources;
    stl::vector<ResourceHandle>     vacant;
    Spec                            spec;
    HWND    hwnd;
    HDC     hdc;
    HGLRC   hglrc;

    Members()
        : immediate_context(NULL)
        , hwnd(NULL)
        , hdc(NULL)
        , hglrc(NULL)
    {
    }
};
istMemberPtrImpl_Noncopyable(Device,Members);

HDC Device::getHDC() { return m->hdc; }
HGLRC Device::getHGLRC() { return m->hglrc; }


Device::Device(HWND hwnd)
{
    m->hwnd = hwnd;
    m->hdc = ::GetDC(m->hwnd);
#ifdef i3d_enable_resource_leak_check

#endif // i3d_enable_leak_check

    int pixelformat;
    static PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR),    //この構造体のサイズ
        1,                  //OpenGLバージョン
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,       //ダブルバッファ使用可能
        PFD_TYPE_RGBA,      //RGBAカラー
        32,                 //色数
        0, 0,               //RGBAのビットとシフト設定        
        0, 0,                //G
        0, 0,                //B
        0, 0,                //A
        0,                  //アキュムレーションバッファ
        0, 0, 0, 0,         //RGBAアキュムレーションバッファ
        32,                 //Zバッファ    
        0,                  //ステンシルバッファ
        0,                  //使用しない
        PFD_MAIN_PLANE,     //レイヤータイプ
        0,                  //予約
        0, 0, 0             //レイヤーマスクの設定・未使用
    };

    // glew 用の仮のコンテキスト生成
    if(((pixelformat = ::ChoosePixelFormat(m->hdc, &pfd)) == 0)
        || ((::SetPixelFormat(m->hdc, pixelformat, &pfd) == FALSE))
        || (!(m->hglrc=::wglCreateContext(m->hdc))))
    {
            istPrint("OpenGL initialization failed");
    }
    wglMakeCurrent(m->hdc, m->hglrc);
    glewInit();
    {
        const GLubyte *version = glGetString(GL_VERSION);
        const GLubyte *vendor = glGetString(GL_VENDOR);
        istPrint("OpenGL version: %s, vendor: %s\n", version, vendor);
        if(strcmp((const char*)vendor, "Intel")==0) { m->spec.needs_transpose=true; }
    }
}

Device::~Device()
{
    for(uint32 i=0; i<m->resources.size(); ++i) {
        istSafeRelease(m->resources[i]);
    }
    m->resources.clear();

    if(m->hglrc!=NULL) {
        ::wglMakeCurrent(NULL, NULL);
        ::wglDeleteContext(m->hglrc);
        m->hglrc = NULL;
    }
    if(m->hdc!=NULL) {
        ::ReleaseDC(m->hwnd, m->hdc);
        m->hdc = NULL;
    }
    g_the_device = NULL;
}
#endif // ist_env_Windows

DeviceContext* Device::getImmediateContext()
{
    return m->immediate_context;
}

const Spec* Device::getSpec() const
{
    return &m->spec;
}

void Device::addResource( DeviceResource *v )
{
    if(!v) { return; }

    if(!m->vacant.empty()) {
        ResourceHandle drh = m->vacant.back();
        m->vacant.pop_back();
        m->resources[drh] = v;
        v->setDeviceResourceHandle(drh);
    }
    else {
        v->setDeviceResourceHandle(m->resources.size());
        m->resources.push_back(v);
    }
}

void Device::deleteResource( ResourceHandle v )
{
    istSafeDelete(m->resources[v]);
    m->vacant.push_back(v);
}

DeviceContext* Device::createImmediateContext()
{
    DeviceContext *r = istNew(DeviceContext)(this);
    istAssert(m->immediate_context==NULL);
    m->immediate_context = r;
    return r;
}

Buffer* Device::createBuffer(const BufferDesc &desc)
{
    Buffer *r = istNew(Buffer)(this, desc);
    addResource(r);
    return r;
}

VertexArray* Device::createVertexArray()
{
    VertexArray *r = istNew(VertexArray)(this);
    addResource(r);
    return r;
}

VertexShader* Device::createVertexShader(const VertexShaderDesc &desc)
{
    VertexShader *r = istNew(VertexShader)(this, desc);
    addResource(r);
    return r;
}

PixelShader* Device::createPixelShader(const PixelShaderDesc &desc)
{
    PixelShader *r = istNew(PixelShader)(this, desc);
    addResource(r);
    return r;
}

GeometryShader* Device::createGeometryShader(const GeometryShaderDesc &desc)
{
    GeometryShader *r = istNew(GeometryShader)(this, desc);
    addResource(r);
    return r;
}

ShaderProgram* Device::createShaderProgram(const ShaderProgramDesc &desc)
{
    ShaderProgram *r = istNew(ShaderProgram)(this, desc);
    addResource(r);
    return r;
}

Sampler* Device::createSampler(const SamplerDesc &desc)
{
    Sampler *r = istNew(Sampler)(this, desc);
    addResource(r);
    return r;
}

Texture1D* Device::createTexture1D(const Texture1DDesc &desc)
{
    Texture1D *r = istNew(Texture1D)(this, desc);
    addResource(r);
    return r;
}

Texture2D* Device::createTexture2D(const Texture2DDesc &desc)
{
    Texture2D *r = istNew(Texture2D)(this, desc);
    addResource(r);
    return r;
}

Texture3D* Device::createTexture3D(const Texture3DDesc &desc)
{
    Texture3D *r = istNew(Texture3D)(this, desc);
    addResource(r);
    return r;
}

RenderTarget* Device::createRenderTarget()
{
    RenderTarget *r = istNew(RenderTarget)(this);
    addResource(r);
    return r;
}

BlendState* Device::createBlendState( const BlendStateDesc &desc )
{
    BlendState *r = istNew(BlendState)(this, desc);
    addResource(r);
    return r;
}

DepthStencilState* Device::createDepthStencilState( const DepthStencilStateDesc &desc )
{
    DepthStencilState *r = istNew(DepthStencilState)(this, desc);
    addResource(r);
    return r;
}

void Device::swapBuffers()
{
    ::SwapBuffers(m->hdc);
}


#ifdef i3d_enable_resource_leak_check
void Device::printLeakInfo()
{
    for(size_t i=0; i<m->resources.size(); ++i) {
        if(m->resources[i]==NULL) { continue; }
        istPrint("i3dgl::Device: resource leak %p\n", m->resources[i]);
        m->resources[i]->printLeakInfo();
        istPrint("\n");
    }
}

#endif // i3d_enable_leak_check


Device* GetDevice()
{
    return g_the_device;
}

Device* CreateDevice( HWND hwnd )
{
    if(!g_the_device) {
        g_the_device = istNew(Device)(hwnd);
    }
    return g_the_device;
}

} // namespace i3d
} // namespace ist
#endif // ist_with_OpenGL
