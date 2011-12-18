#include "stdafx.h"
#include "../Base/Assert.h"
#include "../Base/TaskScheduler.h"
#include "../Sound.h"
#include "../Window.h"

namespace ist
{

Application *g_the_app = NULL;
HINSTANCE g_hinstance = NULL;

#ifdef IST_OPENGL
HDC g_hdc = NULL;
#endif // IST_OPENGL


LRESULT CALLBACK WndProc(HWND hwnd , UINT message , WPARAM wParam , LPARAM lParam)
{
    Application *app = g_the_app;

    static HIMC himc;
    static wchar_t imebuf[512];

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


    case WM_IME_SETCONTEXT:
        himc = ImmGetContext(app->getWindowHandle());
        return ::DefWindowProc(hwnd, message, wParam, lParam);
        break;

    case WM_IME_CHAR:
        return ::DefWindowProc(hwnd, message, wParam, lParam);
        break;

    case WM_IME_COMPOSITION:
        if(lParam & GCS_RESULTSTR) {
            WM_IME wm; wm.initialize();
            wm.type = WindowMessage::MES_IME_RESULT;
            wm.text_len = ::ImmGetCompositionString(himc, GCS_RESULTSTR, imebuf, _countof(imebuf)) / sizeof(wchar_t);
            if(wm.text_len!=_countof(imebuf)) { imebuf[wm.text_len]=L'\0'; }
            wm.text = imebuf;
            app->handleWindowMessage(wm);
        }
        if(lParam & GCS_COMPSTR) {
            WM_IME wm; wm.initialize();
            wm.type = WindowMessage::MES_IME_CHAR;
            wm.text_len = ::ImmGetCompositionString(himc, GCS_COMPSTR, imebuf, _countof(imebuf)) / sizeof(wchar_t);
            if(wm.text_len!=_countof(imebuf)) { imebuf[wm.text_len]=L'\0'; }
            wm.text = imebuf;
            app->handleWindowMessage(wm);
        }
        if(lParam & GCS_CURSORPOS) {
            WM_IME wm; wm.initialize();
            wm.type = WindowMessage::MES_IME_CURSOR_MOVE;
            wm.cursor_pos = ::ImmGetCompositionString(himc, GCS_CURSORPOS, imebuf, _countof(imebuf));
            app->handleWindowMessage(wm);
        }
        return ::DefWindowProc(hwnd, message, wParam, lParam);
        break;

    case WM_IME_CONTROL:
        return ::DefWindowProc(hwnd, message, wParam, lParam);
        break;

    case WM_IME_NOTIFY:
        {
            WM_IME wm; wm.initialize();
            wm.type = WindowMessage::MES_IME_CHAR;
            wm.text_len = ::ImmGetCompositionString(himc, GCS_COMPSTR, imebuf, _countof(imebuf));
            wm.text = imebuf;
            app->handleWindowMessage(wm);
        }
        return ::DefWindowProc(hwnd, message, wParam, lParam);
        break;

    case WM_IME_STARTCOMPOSITION:
        {
            WM_IME wm; wm.initialize();
            wm.type = WindowMessage::MES_IME_BEGIN;
            app->handleWindowMessage(wm);
        }
        return ::DefWindowProc(hwnd, message, wParam, lParam);
        break;

    case WM_IME_ENDCOMPOSITION:
        {
            WM_IME wm; wm.initialize();
            wm.type = WindowMessage::MES_IME_END;
            app->handleWindowMessage(wm);
        }
        return ::DefWindowProc(hwnd, message, wParam, lParam);
        break;


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


    case WM_SIZE:
        {
            WM_WindowSize wm;
            wm.type = WindowMessage::MES_WINDOW_SIZE;
            wm.window_size.x = LOWORD(lParam);
            wm.window_size.y = HIWORD(lParam);
            app->handleWindowMessage(wm);
        }
        return 0;

    case WM_MOVE:
        {
            WM_WindowMove wm;
            wm.type = WindowMessage::MES_WINDOW_MOVE;
            wm.window_pos.x = LOWORD(lParam);
            wm.window_pos.y = HIWORD(lParam);
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

            return ::DefWindowProc(hwnd, message, wParam, lParam);
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



Application* Application::getInstance()
{
    return g_the_app;
}

Application::Application()
: m_hwnd(NULL)
, m_width(0)
, m_height(0)
, m_graphics_error(ERR_NOERROR)
#if defined(IST_OPENGL) && defined(WIN32)
, m_hdc(NULL)
, m_hglrc(NULL)
#endif // defined(IST_OPENGL) && defined(WIN32)
#ifdef IST_DIRECTX
, m_dxswapchain(0)
, m_dxdevice(0)
, m_dxcontext(0)
#endif // IST_DIRECTX
#ifdef IST_OPENCL
, m_cl_context(NULL)
, m_cl_queue(NULL)
#endif // IST_OPENCL
{
}


Application::~Application()
{
    if(g_the_app==this) { g_the_app=NULL; }
}

bool Application::initialize(ivec2 wpos, ivec2 wsize, const wchar_t *title, bool fullscreen)
{
    if(g_the_app) {
        IST_PRINT("既にインスタンスが存在している");
        return false;
    }
    g_the_app = this;

    m_width = wsize.x;
    m_height = wsize.y;
    m_fullscreen = fullscreen;
    int style = fullscreen ? WS_POPUP : WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME ^ WS_MAXIMIZEBOX;
    int flag = WS_POPUP | WS_VISIBLE;

    RECT rect = {0, 0, wsize.x, wsize.y};
    ::AdjustWindowRect(&rect, style, FALSE);
    wsize.x = rect.right - rect.left;
    wsize.y = rect.bottom - rect.top;

    WNDCLASSEX wc;
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
    if(::RegisterClassEx(&wc)==NULL) {
        IST_PRINT("RegisterClassEx() failed");
        return false;
    }

    m_hwnd = ::CreateWindow(title, title, style, wpos.x,wpos.y, wsize.x,wsize.y, NULL, NULL, g_hinstance, NULL);
    if(m_hwnd==NULL) {
        IST_PRINT("CreateWindow() failed");
        return false;
    }

    DEVMODE devmode_sav;
    if(m_fullscreen) {
        devmode_sav.dmSize = sizeof(devmode_sav);
        devmode_sav.dmDriverExtra = 0;
        devmode_sav.dmPelsWidth = wsize.x;
        devmode_sav.dmPelsHeight = wsize.y;
        HDC hdc = GetDC(NULL);
        devmode_sav.dmBitsPerPel = GetDeviceCaps(hdc,BITSPIXEL);
        ReleaseDC(0,hdc);
        devmode_sav.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT;

        ::SetWindowPos(m_hwnd,(HWND)-1, 0,0, wsize.x,wsize.y, SWP_SHOWWINDOW);
        if(::ChangeDisplaySettings(&devmode_sav,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL) {
            return false;
        }
    }
    else {
        ::SetWindowPos(m_hwnd,(HWND)-1, wpos.x,wpos.y, wsize.x,wsize.y, SWP_SHOWWINDOW);
    }

    return true;
}

void Application::finalize()
{
#ifdef IST_OPENCL
    if(m_cl_context) { delete m_cl_context; m_cl_context=NULL; }
#endif // IST_OPENCL
    if(m_hwnd) {
        ::CloseWindow(m_hwnd);
        m_hwnd=NULL;
    }
}

bool Application::initializeDraw()
{
#if defined(IST_OPENGL) && defined(WIN32)
    m_hdc = ::GetDC(m_hwnd);
    g_hdc = m_hdc;

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
            IST_PRINT("OpenGL initialization failed");
    }
    wglMakeCurrent(m_hdc, m_hglrc);
    glewInit();
    {
        const GLubyte *version = glGetString(GL_VERSION);
        const GLubyte *vendor = glGetString(GL_VENDOR);
        IST_PRINT("OpenGL version: %s, vendor: %s\n", version, vendor);
    }

    //::ShowCursor(false);

    wglSwapIntervalEXT(GL_FALSE);
#endif // IST_OPENGL

#ifdef IST_DIRECTX
    // create a struct to hold information about the swap chain
    DXGI_SWAP_CHAIN_DESC scd;
    // clear out the struct for use
    ZeroMemory(&scd, sizeof(scd));

    // fill the swap chain description struct
    scd.BufferCount = 1;                                    // one back buffer
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;     // use 32-bit color
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;      // how swap chain is to be used
    scd.OutputWindow = m_hwnd;                              // the window to be used
    scd.SampleDesc.Count = 4;                               // how many multisamples
    scd.Windowed = m_fullscreen ? FALSE : TRUE;             // windowed/full-screen mode

    // create a device, device context and swap chain using the information in the scd struct
    D3D11CreateDeviceAndSwapChain(
        NULL,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        NULL,
        NULL,
        NULL,
        D3D11_SDK_VERSION,
        &scd,
        &m_dxswapchain,
        &m_dxdevice,
        NULL,
        &m_dxcontext);
#endif // IST_DIRECTX

#ifdef IST_OPENCL
    // initialize OpenCL
    {
        cl_int err;
        std::vector< cl::Platform > platforms;
        cl::Platform::get(&platforms);
        if(platforms.empty()) {
            IST_PRINT("OpenCL initialization failed");
            return false;
        }

        std::string version;
        std::string vendor;
        platforms[0].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &version);
        platforms[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &vendor);
        IST_PRINT("OpenCL version: %s, vendor: %s\n", version.c_str(), vendor.c_str());
 
        cl_context_properties cprops[3] =  {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

        m_cl_context = new cl::Context(CL_DEVICE_TYPE_DEFAULT,  cprops, NULL, NULL, &err);
        if(err != CL_SUCCESS) {
            IST_PRINT("OpenCL create context failed. error code: 0x%08x\n", err);
        }
        std::vector<cl::Device> cl_devices = m_cl_context->getInfo<CL_CONTEXT_DEVICES>();
        m_cl_queue = new cl::CommandQueue(*m_cl_context, cl_devices[0], 0, &err);
        if(err != CL_SUCCESS) {
            IST_PRINT("OpenCL create command queue failed. error code: 0x%08x\n", err);
        }
    }
#endif // IST_OPENCL


    // CUDA
    {
        cudaError_t e;
        int dev_count;
        e = cudaGetDeviceCount(&dev_count);
        if(e==cudaErrorNoDevice) {
            m_graphics_error = ERR_CUDA_NO_DEVICE;
            return false;
        }
        else if(e==cudaErrorInsufficientDriver) {
            m_graphics_error = ERR_CUDA_INSUFFICIENT_DRIVER;
            return false;
        }

        int device_id = cutGetMaxGflopsDeviceId();
        CUDA_SAFE_CALL( cudaSetDevice(device_id) );
        CUDA_SAFE_CALL( cudaGLSetGLDevice(device_id) );
    }
    return true;
}

void Application::finalizeDraw()
{
#if defined(IST_OPENGL) && defined(WIN32)
    if(m_hglrc!=NULL) {
        ::wglMakeCurrent(NULL, NULL);
        ::wglDeleteContext(m_hglrc);
        m_hglrc = NULL;
    }
    if(m_hdc!=NULL) {
        ::ReleaseDC(m_hwnd, m_hdc);
        m_hdc = NULL;
    }
#endif // defined(IST_OPENGL) && defined(WIN32)

#ifdef IST_DIRECTX
    if(m_dxswapchain)   { m_dxswapchain->Release(); m_dxswapchain=NULL; }
    if(m_dxdevice)      { m_dxdevice->Release();    m_dxdevice=NULL;    }
    if(m_dxcontext)     { m_dxcontext->Release();   m_dxcontext=NULL;   }
#endif // IST_DIRECTX
}

void Application::updateInput()
{
    // keyboard
    ::GetKeyboardState( m_keyboard_state.getRawKeyState() );

    // mouse
    {
        CURSORINFO cinfo;
        cinfo.cbSize = sizeof(cinfo);
        ::GetCursorInfo(&cinfo);
        m_mouse_state.setX( cinfo.ptScreenPos.x );
        m_mouse_state.setX( cinfo.ptScreenPos.y );
    }
    {
        short mouse_button = 0;
        mouse_button |= m_keyboard_state.isKeyPressed(VK_LBUTTON) ? MouseState::BU_LEFT : 0;
        mouse_button |= m_keyboard_state.isKeyPressed(VK_RBUTTON) ? MouseState::BU_RIGHT : 0;
        mouse_button |= m_keyboard_state.isKeyPressed(VK_MBUTTON) ? MouseState::BU_MIDDLE : 0;
        m_mouse_state.setButtonState(mouse_button);
    }

    // joystick
    size_t num_joysticks = stl::min<size_t>(::joyGetNumDevs(), MAX_JOYSTICK_NUM);
    for(size_t i=0; i<num_joysticks; ++i) {
        JOYINFOEX joyinfo;//ジョイスティック情報
        joyinfo.dwSize = sizeof(JOYINFOEX);
        joyinfo.dwFlags = JOY_RETURNALL;
        if(::joyGetPosEx(i, &joyinfo)==JOYERR_NOERROR){
            m_joy_state->setValue(joyinfo);
        }
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

int Application::showMessageDialog( const wchar_t* message, const wchar_t* caption, int dlgtype )
{
    return ::MessageBox(m_hwnd, message, caption, dlgtype);
}

DisplaySetting Application::getCurrentDisplaySetting() const
{
    DEVMODE mode;
    if(::EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &mode)) {
        return DisplaySetting(ivec2(mode.dmPelsWidth, mode.dmPelsHeight), mode.dmBitsPerPel, mode.dmDisplayFrequency);
    }
    return DisplaySetting();
}

void Application::getAvalableDisplaySettings( DisplaySetting*& settings, int& num_settings ) const
{
    static std::vector<DisplaySetting> dsv;
    if(dsv.empty()) {
        int i = 0;
        DEVMODE mode;
        while(::EnumDisplaySettings(NULL, i++, &mode)) {
            dsv.push_back( DisplaySetting(ivec2(mode.dmPelsWidth, mode.dmPelsHeight), mode.dmBitsPerPel, mode.dmDisplayFrequency) );
        }
    }
    settings = &dsv[0];
    num_settings = dsv.size();
}

} // namespace ist

void glSwapBuffers()
{
#ifdef IST_OPENGL
    ::SwapBuffers(ist::g_hdc);
#endif // IST_OPENGL
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
