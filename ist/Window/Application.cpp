#include "istPCH.h"
#include "ist/Base.h"
#include "ist/Debug.h"
#include "ist/Window.h"
#include "Windowsx.h"

namespace ist
{

Application *g_the_app = NULL;
HINSTANCE g_hinstance = NULL;


inline void SetupWMMouse(WM_Mouse &wm, WPARAM wParam , LPARAM lParam, bool screen_to_client=false)
{
    if(wParam&MK_LBUTTON) { wm.button.left=1; }
    if(wParam&MK_RBUTTON) { wm.button.right=1; }
    if(wParam&MK_MBUTTON) { wm.button.middle=1; }
    if(wParam&MK_CONTROL) { wm.button.ctrl=1; }
    if(wParam&MK_SHIFT)   { wm.button.shift=1; }
    wm.wheel = GET_WHEEL_DELTA_WPARAM(wParam);
    POINT pos = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
    if(screen_to_client) {
        ::ScreenToClient(istGetAplication()->getWindowHandle(), &pos);
    }
    wm.mouse_pos.x = (float32)pos.x;
    wm.mouse_pos.y = (float32)pos.y;
}


LRESULT CALLBACK istWndProc(HWND hwnd , UINT message , WPARAM wParam , LPARAM lParam)
{
    Application *app = g_the_app;

    static HIMC himc;
    static wchar_t imebuf[512];

    switch(message)
    {
    case WM_CREATE:
        {
            ::timeBeginPeriod(1);

            WM_Window wm;
            wm.type = WMT_WindowOpen;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_CLOSE:
        {
            WM_Window wm;
            wm.type = WMT_WindowClose;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_DESTROY:
        {
            ::timeEndPeriod(1);
            ::PostQuitMessage(0);
        }
        break;

    case WM_SIZE:
        {
            WM_Window wm;
            wm.type = WMT_WindowSize;
            wm.window_size.x = LOWORD(lParam);
            wm.window_size.y = HIWORD(lParam);
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_MOVE:
        {
            WM_Window wm;
            wm.type = WMT_WindowMove;
            wm.window_pos.x = LOWORD(lParam);
            wm.window_pos.y = HIWORD(lParam);
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_ACTIVATEAPP:
        {
            WM_Window wm;
            wm.type = wParam==TRUE ? WMT_WindowFocus : WMT_WindowDefocus;
            app->_handleWindowMessage(wm);
        }
        break;



    case WM_KEYDOWN:    // fall through
    case WM_KEYUP:      // 
    case WM_CHAR:       // 
        {
            WM_Keyboard wm;
            switch(message) {
            case WM_KEYDOWN:    wm.type=WMT_KeyDown; break;
            case WM_KEYUP:      wm.type=WMT_KeyUp; break;
            case WM_CHAR:       wm.type=WMT_KeyChar; break;
            }
            wm.key = (short)wParam;
            app->_handleWindowMessage(wm);
        }
        break;


    case WM_LBUTTONDOWN:    // fall through
    case WM_RBUTTONDOWN:    // 
    case WM_MBUTTONDOWN:    // 
        {
            WM_Mouse wm;
            wm.type = WMT_MouseDown;
            SetupWMMouse(wm, wParam, lParam);
            if(message==WM_LBUTTONDOWN) wm.button.left=1;
            if(message==WM_RBUTTONDOWN) wm.button.right=1;
            if(message==WM_MBUTTONDOWN) wm.button.middle=1;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_LBUTTONUP:      // fall through
    case WM_RBUTTONUP:      // 
    case WM_MBUTTONUP:      // 
        {
            WM_Mouse wm;
            wm.type = WMT_MouseUp;
            SetupWMMouse(wm, wParam, lParam);
            if(message==WM_LBUTTONUP) wm.button.left=1;
            if(message==WM_RBUTTONUP) wm.button.right=1;
            if(message==WM_MBUTTONUP) wm.button.middle=1;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_MOUSEMOVE:
        {
            WM_Mouse wm;
            wm.type = WMT_MouseMove;
            SetupWMMouse(wm, wParam, lParam);
            static vec2 s_prev_pos = wm.mouse_pos;
            wm.mouse_move = wm.mouse_pos - s_prev_pos;
            s_prev_pos = wm.mouse_pos;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_MOUSEWHEEL:
        {
            WM_Mouse wm;
            SetupWMMouse(wm, wParam, lParam, true);
            wm.type = wm.wheel<0 ? WMT_MouseWheelDown : WMT_MouseWheelUp;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_MOUSEHWHEEL:
        {
            WM_Mouse wm;
            SetupWMMouse(wm, wParam, lParam, true);
            wm.type = wm.wheel<0 ? WMT_MouseWheelLeft : WMT_MouseWheelRight;
            app->_handleWindowMessage(wm);
        }
        break;



    case WM_IME_SETCONTEXT:
        himc = ImmGetContext(app->getWindowHandle());
        break;

    case WM_IME_CHAR:
        break;

    case WM_IME_COMPOSITION:
        if(lParam & GCS_RESULTSTR) {
            WM_IME wm;
            wm.type = WMT_IMEResult;
            wm.text_len = ::ImmGetCompositionString(himc, GCS_RESULTSTR, imebuf, _countof(imebuf)) / sizeof(wchar_t);
            if(wm.text_len!=_countof(imebuf)) { imebuf[wm.text_len]=L'\0'; }
            wm.text = imebuf;
            app->_handleWindowMessage(wm);
        }
        if(lParam & GCS_COMPSTR) {
            WM_IME wm;
            wm.type = WMT_IMEChar;
            wm.text_len = ::ImmGetCompositionString(himc, GCS_COMPSTR, imebuf, _countof(imebuf)) / sizeof(wchar_t);
            if(wm.text_len!=_countof(imebuf)) { imebuf[wm.text_len]=L'\0'; }
            wm.text = imebuf;
            app->_handleWindowMessage(wm);
        }
        if(lParam & GCS_CURSORPOS) {
            WM_IME wm;
            wm.type = WMT_IMECursorMove;
            wm.cursor_pos = ::ImmGetCompositionString(himc, GCS_CURSORPOS, imebuf, _countof(imebuf));
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_IME_CONTROL:
        break;

    case WM_IME_NOTIFY:
        {
            WM_IME wm;
            wm.type = WMT_IMENotify;
            wm.text_len = ::ImmGetCompositionString(himc, GCS_COMPSTR, imebuf, _countof(imebuf));
            wm.text = imebuf;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_IME_STARTCOMPOSITION:
        {
            WM_IME wm;
            wm.type = WMT_IMEBegin;
            app->_handleWindowMessage(wm);
        }
        break;

    case WM_IME_ENDCOMPOSITION:
        {
            WM_IME wm;
            wm.type = WMT_IMEEnd;
            app->_handleWindowMessage(wm);
        }
        break;
    }
    return ::DefWindowProc(hwnd, message, wParam, lParam);
}


struct Application::Members
{
    HWND        hwnd;
    DEVMODE     devmode;
    bool        fullscreen;

    KeyboardState   keyboard_state;
    MouseState      mouse_state;
    JoyState        joy_state[Application::MAX_JOYSTICK_NUM];

    uvec2   window_size;
    ist::vector<WMHandler*> wmhandlers;

    Members()
        : hwnd(NULL)
        , fullscreen(false)
    {
    }
};
istMemberPtrImpl(Application,Members);



Application* Application::getInstance()
{
    return g_the_app;
}

Application::Application()
{
}


Application::~Application()
{
    if(g_the_app==this) { g_the_app=NULL; }
}


bool Application::isFullscreen() const                      { return m->fullscreen; }
const uvec2& Application::getWindowSize() const             { return m->window_size; }
const KeyboardState& Application::getKeyboardState() const  { return m->keyboard_state; }
const MouseState& Application::getMouseState() const        { return m->mouse_state; }
const JoyState& Application::getJoyState(int i) const       { return m->joy_state[i]; }
HWND Application::getWindowHandle() const                   { return m->hwnd; }


bool Application::initialize(ivec2 wpos, ivec2 wsize, const wchar_t *title, bool fullscreen)
{
    if(g_the_app) {
        istPrint("既にインスタンスが存在している");
        return false;
    }
    g_the_app = this;

    m->window_size = wsize;
    m->fullscreen = fullscreen;
    int style = fullscreen ? WS_POPUP : WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME ^ WS_MAXIMIZEBOX;
    int flag = WS_POPUP | WS_VISIBLE;

    RECT rect = {0, 0, wsize.x, wsize.y};
    ::AdjustWindowRect(&rect, style, FALSE);
    wsize.x = rect.right - rect.left;
    wsize.y = rect.bottom - rect.top;

    WNDCLASSEX wc;
    wc.cbSize        = sizeof(wc);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = &istWndProc;
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

    m->hwnd = ::CreateWindow(title, title, style, wpos.x,wpos.y, wsize.x,wsize.y, NULL, NULL, g_hinstance, NULL);
    if(m->hwnd==NULL) {
        istPrint("CreateWindow() failed");
        return false;
    }

    DEVMODE devmode_sav;
    if(m->fullscreen) {
        devmode_sav.dmSize = sizeof(devmode_sav);
        devmode_sav.dmDriverExtra = 0;
        devmode_sav.dmPelsWidth = wsize.x;
        devmode_sav.dmPelsHeight = wsize.y;
        HDC hdc = GetDC(NULL);
        devmode_sav.dmBitsPerPel = GetDeviceCaps(hdc,BITSPIXEL);
        ReleaseDC(0,hdc);
        devmode_sav.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT;

        ::SetWindowPos(m->hwnd,0, 0,0, wsize.x,wsize.y, SWP_SHOWWINDOW);
        if(::ChangeDisplaySettings(&devmode_sav,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL) {
            return false;
        }
    }
    else {
        ::SetWindowPos(m->hwnd,0, wpos.x,wpos.y, wsize.x,wsize.y, SWP_SHOWWINDOW);
    }

    for(uint32 i=0; i<0; ++i) {

    }

    return true;
}

void Application::finalize()
{
    if(m->hwnd) {
        ::DestroyWindow(m->hwnd);
        m->hwnd=NULL;
    }
}


void Application::updateInput()
{
    // keyboard
    m->keyboard_state.copyToBack();
    ::GetKeyboardState( m->keyboard_state.getRawKeyState() );

    // mouse
    {
        CURSORINFO cinfo;
        cinfo.cbSize = sizeof(cinfo);
        ::GetCursorInfo(&cinfo);
        m->mouse_state.setX( cinfo.ptScreenPos.x );
        m->mouse_state.setY( cinfo.ptScreenPos.y );
    }
    {
        short mouse_button = 0;
        mouse_button |= m->keyboard_state.isKeyPressed(VK_LBUTTON) ? MouseState::BU_LEFT : 0;
        mouse_button |= m->keyboard_state.isKeyPressed(VK_RBUTTON) ? MouseState::BU_RIGHT : 0;
        mouse_button |= m->keyboard_state.isKeyPressed(VK_MBUTTON) ? MouseState::BU_MIDDLE : 0;
        m->mouse_state.setButtonState(mouse_button);
    }

    // joystick
    for(size_t i=0; i<MAX_JOYSTICK_NUM; ++i) {
        JOYINFOEX joyinfo;
        joyinfo.dwSize = sizeof(JOYINFOEX);
        joyinfo.dwFlags = JOY_RETURNALL;
        if(::joyGetPosEx(i, &joyinfo)==JOYERR_NOERROR){
            m->joy_state->setValue(joyinfo);
        }
    }
}

void Application::translateMessage()
{
    MSG msg;
    while(::PeekMessage(&msg, m->hwnd, 0,0,PM_REMOVE)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
    }
}

int Application::showMessageDialog( const char* message, const char* caption, int dlgtype/*=DLG_OK*/ )
{
    return ::MessageBoxA(m->hwnd, message, caption, dlgtype);
}

int Application::showMessageDialog( const wchar_t* message, const wchar_t* caption, int dlgtype )
{
    return ::MessageBoxW(m->hwnd, message, caption, dlgtype);
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
    static ist::vector<DisplaySetting> dsv;
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

void Application::addMessageHandler( WMHandler *wmh )
{
    m->wmhandlers.push_back(wmh);
}

void Application::eraseMessageHandler( WMHandler *wmh )
{
    m->wmhandlers.erase(
        stl::find(m->wmhandlers.begin(), m->wmhandlers.end(), wmh));
}

bool Application::_handleWindowMessage( const WM_Base& wm )
{
    for(int32 i=m->wmhandlers.size()-1; i>=0; --i) {
        if((*m->wmhandlers[i])(wm)) {
            return true;
        }
    }
    return false;
}

} // namespace ist


namespace {

    void PreMain()
    {
        ::setlocale(LC_ALL, "");
#ifndef ist_env_Master
        ist::InitializeDebugSymbol();
#endif // ist_env_Master
    }

    void PostMain()
    {
        // 他のモジュールがまだシンボル情報を使う可能性があるので敢えて呼ばない
//#ifndef ist_env_Master
//        ist::FinalizeDebugSymbol();
//#endif // ist_env_Master
    }

} // namespace


extern int istmain(int argc, char* argv[]);

int __cdecl main(int argc, char* argv[])
{
    PreMain();
    int r = istmain(argc, argv);
    PostMain();
    return r;
}

#ifdef ist_env_Windows
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prev, LPSTR cmd, int show)
{
    ist::g_hinstance = hInstance;
    PreMain();
    int r = istmain(__argc, __argv);
    PostMain();
    return r;
}
#endif // ist_env_Windows
