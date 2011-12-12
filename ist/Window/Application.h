#ifndef __ist_Application__
#define __ist_Application__

#include "WindowMessage.h"
#include "InputState.h"

void glSwapBuffers();

namespace ist {


class Application
{
public:
    enum DLG_TYPE {
        DLG_OK              = 0x00000000L,
        DLG_OKCANCEL        = 0x00000001L,
        DLG_ABORTIGNORE     = 0x00000002L,
        DLG_CANCELCONTINUE  = 0x00000006L,
        DLG_RETRYCANCEL     = 0x00000005L,
        DLG_YESNO           = 0x00000004L,
        DLG_YESNOCANCEL     = 0x00000003L,
    };
    enum DLG_ICON {
        ICON_EXCLAMATION    = 0x00000030L,
        ICON_WARNING        = 0x00000030L,
        ICON_INFORMATION    = 0x00000040L,
        ICON_ASTERISK       = 0x00000040L,
        ICON_QUESTION       = 0x00000020L,
        ICON_ERROR          = 0x00000010L,
        ICON_HAND           = 0x00000010L,
    };
    enum DLG_RET {
        DLGRET_OK       = 1,
        DLGRET_CANCEL   = 2,
        DLGRET_ABORT    = 3,
        DLGRET_RETRY    = 4,
        DLGRET_IGNORE   = 5,
        DLGRET_YES      = 6,
        DLGRET_NO       = 7,
        DLGRET_TRYAGAIN = 10,
        DLGRET_CONTINUE = 11,
    };
    enum ERROR_CODE {
        ERR_NOERROR,
        ERR_CREATEWINDOW_FAILED,
        ERR_CHANGEDISPLAYSETTINGS_FAILED,
        ERR_OPENAL_OPENDEVICE_FAILED,
        ERR_OPENAL_CREATECONTEXT_FAILED,
        ERR_OPENGL_330_IS_NOT_SUPPORTED,
        ERR_OPENGL_INITIALIZATION_FAILED,
        ERR_DIRECT3D11_INITIALIZATION_FAILED,
        ERR_CUDA_NO_DEVICE,
        ERR_CUDA_INSUFFICIENT_DRIVER,
    };

private:
    static const int MAX_JOYSTICK_NUM = 4;

    HWND        m_hwnd;
    DEVMODE     m_devmode;
    bool        m_fullscreen;

    KeyboardState   m_keyboard_state;
    MouseState      m_mouse_state;
    JoyState        m_joy_state[MAX_JOYSTICK_NUM];

    size_t m_width, m_height;
    ERROR_CODE m_graphics_error;

#if defined(IST_OPENGL) && defined(WIN32)
    HDC         m_hdc;
    HGLRC       m_hglrc;
#endif // IST_OPENGL WIN32

#ifdef IST_DIRECTX
    IDXGISwapChain      *m_dxswapchain;
    ID3D11Device        *m_dxdevice;
    ID3D11DeviceContext *m_dxcontext;
#endif // IST_DIRECTX

#ifdef IST_OPENCL
    cl::Context *m_cl_context;
    cl::CommandQueue *m_cl_queue;
#endif // IST_OPENCL


public:
    static Application* getInstance();

    Application();
    virtual ~Application();

    virtual bool initialize(ivec2 wpos, ivec2 wsize, const wchar_t *title, bool fullscreen=false);
    virtual void finalize();

    virtual bool initializeDraw();
    virtual void finalizeDraw();

    virtual void mainLoop()=0;
    virtual int handleWindowMessage(const WindowMessage& wm)=0;

    virtual void updateInput();
    virtual void translateMessage();

    int showMessageDialog(const wchar_t* message, const wchar_t* caption, int dlgtype=DLG_OK);

    size_t getWindowWidth() const   { return m_width; }
    size_t getWindowHeight() const  { return m_height; }
    const KeyboardState& getKeyboardState() const   { return m_keyboard_state; }
    const MouseState& getMouseState() const         { return m_mouse_state; }
    const JoyState& getJoyState(int i=0) const      { return m_joy_state[i]; }
    ERROR_CODE getGraphicsError() const             { return m_graphics_error; }

    DisplaySetting getCurrentDisplaySetting() const;
    void getAvalableDisplaySettings(DisplaySetting*& settings, int& num_settings) const;

#ifdef WIN32
    HWND getWindowHandle() const { return m_hwnd; }
#endif // WIN32
#ifdef IST_OPENCL
    cl::Context* getCLContext() { return m_cl_context; }
    cl::CommandQueue* getCLCommandQueue() { return m_cl_queue; }
#endif // IST_OPENCL
#if defined(IST_OPENGL) && defined(WIN32)
    HDC     getHDC() const { return m_hdc; }
    HGLRC   getHGLRC() const { return m_hglrc; }
#endif // IST_OPENGL WIN32
};

} // namespace ist


#define istGetAplication()  ist::Application::getInstance()
#define istShowMessageDialog(mes, cap, dtype) istGetAplication()->showMessageDialog(mes, cap, dtype)


#endif // __ist_Application__
