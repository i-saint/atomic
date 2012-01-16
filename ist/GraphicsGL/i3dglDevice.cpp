#include "stdafx.h"
#include "../Base.h"
#include "i3dglDevice.h"
#include "i3dglDeviceContext.h"

namespace ist {
namespace i3dgl {

#ifdef _WIN32

Device::Device(HWND hwnd) : m_hwnd(hwnd), m_context(NULL)
{
    m_hdc = ::GetDC(m_hwnd);

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
    if(((pixelformat = ::ChoosePixelFormat(m_hdc, &pfd)) == 0)
        || ((::SetPixelFormat(m_hdc, pixelformat, &pfd) == FALSE))
        || (!(m_hglrc=::wglCreateContext(m_hdc)))) {
            istPrint("OpenGL initialization failed");
    }
    wglMakeCurrent(m_hdc, m_hglrc);
    glewInit();
    {
        const GLubyte *version = glGetString(GL_VERSION);
        const GLubyte *vendor = glGetString(GL_VENDOR);
        istPrint("OpenGL version: %s, vendor: %s\n", version, vendor);
    }

    m_context = istNew(DeviceContext)(this);
}

Device::~Device()
{
    for(uint32 i=0; i<m_resources.size(); ++i) {
        istSafeDelete(m_resources[i]);
    }

    istSafeDelete(m_context);

    if(m_hglrc!=NULL) {
        ::wglMakeCurrent(NULL, NULL);
        ::wglDeleteContext(m_hglrc);
        m_hglrc = NULL;
    }
    if(m_hdc!=NULL) {
        ::ReleaseDC(m_hwnd, m_hdc);
        m_hdc = NULL;
    }
}
#endif // _WIN32


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

ShaderProgram* Device::createShaderProgram()
{
    ShaderProgram *r = istNew(ShaderProgram)(this);
    addResource(r);
    return r;
}

Texture2D* Device::createTexture2D()
{
    Texture2D *r = istNew(Texture2D)(this);
    addResource(r);
    return r;
}

RenderTarget* Device::createRenderTarget()
{
    RenderTarget *r = istNew(RenderTarget)(this);
    addResource(r);
    return r;
}

void Device::swapBuffers()
{
    ::SwapBuffers(m_hdc);
}

} // namespace i3d
} // namespace ist
