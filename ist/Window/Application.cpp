#include "stdafx.h"
#include "../Base.h"
#include "../Sound.h"
#include "../Window.h"

namespace ist
{

Application *g_the_app = NULL;
HINSTANCE g_hinstance = NULL;


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
{
}


Application::~Application()
{
    if(g_the_app==this) { g_the_app=NULL; }
}

bool Application::initialize(ivec2 wpos, ivec2 wsize, const wchar_t *title, bool fullscreen)
{
    if(g_the_app) {
        istPrint("既にインスタンスが存在している");
        return false;
    }
    g_the_app = this;

    m_window_size = wsize;
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
        istPrint("RegisterClassEx() failed");
        return false;
    }

    m_hwnd = ::CreateWindow(title, title, style, wpos.x,wpos.y, wsize.x,wsize.y, NULL, NULL, g_hinstance, NULL);
    if(m_hwnd==NULL) {
        istPrint("CreateWindow() failed");
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

    for(uint32 i=0; i<0; ++i) {

    }

    return true;
}

void Application::finalize()
{
    if(m_hwnd) {
        ::CloseWindow(m_hwnd);
        m_hwnd=NULL;
    }
}


void Application::updateInput()
{
    // keyboard
    m_keyboard_state.copyToBack();
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
    for(size_t i=0; i<MAX_JOYSTICK_NUM; ++i) {
        JOYINFOEX joyinfo;
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

int Application::showMessageDialog( const char* message, const char* caption, int dlgtype/*=DLG_OK*/ )
{
    return ::MessageBoxA(m_hwnd, message, caption, dlgtype);
}

int Application::showMessageDialog( const wchar_t* message, const wchar_t* caption, int dlgtype )
{
    return ::MessageBoxW(m_hwnd, message, caption, dlgtype);
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
    static stl::vector<DisplaySetting> dsv;
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
