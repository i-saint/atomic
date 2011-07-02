#include "stdafx.h"
#include <windows.h>
#include "../Base/Assert.h"
#include "../Base/TaskScheduler.h"
#include "Application.h"

namespace ist
{

Application *g_the_app = NULL;
HINSTANCE g_hinstance = NULL;
HDC g_hdc = NULL;


LRESULT CALLBACK WndProc(HWND hwnd , UINT message , WPARAM wParam , LPARAM lParam)
{
    Application *app = g_the_app;
    switch(message)
    {
    case WM_ACTIVATEAPP:
        {
            WM_Active wm;
            wm.type = WindowMessage::MES_ACTIVE;
            wm.state = wParam==TRUE ? WM_Active::ST_ACTIVATED : WM_Active::ST_DEACTIVATED;
            app->handleWindowMessage(wm);
        }
        return 0;

    case WM_KEYDOWN:    // fall through
    case WM_KEYUP:      // 
    case WM_CHAR:       // 
        {
            WM_Keyboard wm;
            wm.type = WindowMessage::MES_KEYBOARD;
            wm.key = (short)wParam;
            switch(message) {
            case WM_KEYDOWN:    wm.action=WM_Keyboard::ACT_KEYDOWN; break;
            case WM_KEYUP:      wm.action=WM_Keyboard::ACT_KEYUP; break;
            case WM_CHAR:       wm.action=WM_Keyboard::ACT_CHAR; break;
            }
            app->handleWindowMessage(wm);
        }
        return 0;

    case WM_LBUTTONDOWN:    // fall through
    case WM_LBUTTONUP:      // 
    case WM_RBUTTONDOWN:    // 
    case WM_RBUTTONUP:      // 
    case WM_MBUTTONDOWN:    // 
    case WM_MBUTTONUP:      // 
    case WM_MOUSEMOVE:      // 
        {
            WM_Mouse wm;
            wm.type = WindowMessage::MES_MOUSE;
            app->handleWindowMessage(wm);
        }
        return 0;



    case WM_CREATE:
        {
            ::timeBeginPeriod(1);
        }
        return 0;

    case WM_CLOSE:
        {
            WM_Close wm;
            wm.type = WindowMessage::MES_CLOSE;
            app->handleWindowMessage(wm);

            ::SendMessage(hwnd, WM_DESTROY, 0, 0);
        }
        return 0;

    case WM_DESTROY:
        {
            ::timeEndPeriod(1);
            ::PostQuitMessage(0);
        }
        return 0;
    }
    return ::DefWindowProc(hwnd, message, wParam, lParam);
}



Application::Application()
: m_hwnd(NULL)
, m_hglrc(NULL)
, m_x(0)
, m_y(0)
, m_width(0)
, m_height(0)
{
}

Application::~Application()
{
    finalize();
}

bool Application::initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen)
{
    if(g_the_app) {
        IST_PRINT("既にインスタンスが存在している");
        return false;
    }
    g_the_app = this;

    m_width = width;
    m_height = height;
    m_fullscreen = fullscreen;
    int style = fullscreen ? WS_POPUP : WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME ^ WS_MAXIMIZEBOX;
    int flag = WS_POPUP | WS_VISIBLE;
    if(!m_fullscreen) {
        width  += ::GetSystemMetrics(SM_CXDLGFRAME)*2;
        height += ::GetSystemMetrics(SM_CYDLGFRAME)*2 + ::GetSystemMetrics(SM_CYCAPTION);

        m_x = x;
        m_y = y;
    }
    else {
    }

    WNDCLASSEXW wc;
    wc.cbSize        = sizeof(wc);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = &WndProc;
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = g_hinstance;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wc.lpszMenuName  = NULL;
    wc.lpszClassName = title;
    wc.hIconSm       = LoadIcon (NULL, IDI_APPLICATION);
    if(::RegisterClassExW(&wc)==NULL) {
        IST_PRINT("RegisterClassExW() failed");
        return false;
    }

    m_hwnd = ::CreateWindowW(title, title, style, x,y, width,height, NULL, NULL, g_hinstance, NULL);
    if(m_hwnd==NULL) {
        IST_PRINT("CreateWindowW() failed");
        return false;
    }

    DEVMODE devmode_sav;
    if(m_fullscreen) {
        devmode_sav.dmSize = sizeof(devmode_sav);
        devmode_sav.dmDriverExtra = 0;
        devmode_sav.dmPelsWidth = width;
        devmode_sav.dmPelsHeight = height;
        HDC hdc = GetDC(NULL);
        devmode_sav.dmBitsPerPel = GetDeviceCaps(hdc,BITSPIXEL);
        ReleaseDC(0,hdc);
        devmode_sav.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT;

        ::SetWindowPos(m_hwnd,(HWND)-1, 0,0, width,height, SWP_SHOWWINDOW);
        if(::ChangeDisplaySettings(&devmode_sav,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL) {
            return false;
        }
    }
    else {
        ::SetWindowPos(m_hwnd,(HWND)-1, x,y, width,height, SWP_SHOWWINDOW);
    }

    return true;
}

void Application::finalize()
{
    if(g_the_app==this) {
        g_the_app = NULL;
    }

    if(m_hwnd) {
        ::CloseWindow(m_hwnd);
        m_hwnd = NULL;
    }
}

bool Application::initializeDraw()
{
    int pixelformat;
    static PIXELFORMATDESCRIPTOR pfd = {
        sizeof(PIXELFORMATDESCRIPTOR),    //この構造体のサイズ
        1,                  //OpenGLバージョン
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,       //ダブルバッファ使用可能
        PFD_TYPE_RGBA,      //RGBAカラー
        24,                 //色数
        0, 0,               //RGBAのビットとシフト設定        
        0, 0,                //G
        0, 0,                //B
        0, 0,                //A
        0,                  //アキュムレーションバッファ
        0, 0, 0, 0,         //RGBAアキュムレーションバッファ
        24,                 //Zバッファ    
        0,                  //ステンシルバッファ
        0,                  //使用しない
        PFD_MAIN_PLANE,     //レイヤータイプ
        0,                  //予約
        0, 0, 0             //レイヤーマスクの設定・未使用
    };

    m_hdc = ::GetDC(m_hwnd);
    g_hdc = m_hdc;
    //ピクセルフォーマットの指定 //OpenGLレンダリングコンテキストの作成
    if(((pixelformat = ::ChoosePixelFormat(m_hdc, &pfd)) == 0)
        || ((::SetPixelFormat(m_hdc, pixelformat, &pfd) == FALSE))
        || (!(m_hglrc=::wglCreateContext(m_hdc)))) {
            IST_PRINT("initializing opengl failed");
            return false;
    }

    //PixelFormat初期化
    ::wglMakeCurrent(m_hdc, m_hglrc);
    //::ShowCursor(false);

    ::glewInit();

    return true;
}

void Application::finalizeDraw()
{
    if(m_hglrc) {
        ::wglDeleteContext(m_hglrc);
        m_hglrc = NULL;
    }
}

void Application::translateMessage()
{
    MSG msg;
    while(::PeekMessage(&msg, m_hwnd, 0,0,PM_REMOVE)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
    }
}

} // namespace ist

void glSwapBuffers()
{
    ::SwapBuffers(ist::g_hdc);
}



extern int istmain(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    return istmain(argc, argv);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prev, LPSTR cmd, int show)
{
    ist::g_hinstance = hInstance;
    return istmain(__argc, __argv);
}
