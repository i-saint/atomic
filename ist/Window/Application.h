#ifndef __ist_Application__
#define __ist_Application__

void glSwapBuffers();

namespace ist
{

struct WindowMessage
{
    enum TYPE
    {
        MES_CLOSE,
        MES_ACTIVE,
        MES_KEYBOARD,
        MES_MOUSE,
        MES_JOYSTICK,
        MES_WINDOW_SIZE,
        MES_WINDOW_MOVE,
        MES_FOCUS,
    };

    int type;
};

struct WM_Close : public WindowMessage
{
};

struct WM_Active : public WindowMessage
{
    enum STATE
    {
        ST_ACTIVATED,
        ST_DEACTIVATED,
    };
    short state;
};

struct WM_WindowSize : public WindowMessage
{
    ivec2 window_size;
};

struct WM_WindowMove : public WindowMessage
{
    ivec2 window_pos;
};


struct WM_Keyboard : public WindowMessage
{
    enum ACTION
    {
        ACT_KEYUP,
        ACT_KEYDOWN,
        ACT_CHAR,
    };
    enum KEY
    {
        KEY_ESCAPE = VK_ESCAPE,
    };

    short action;
    short key;
};

struct WM_Mouse : public WindowMessage
{
    enum ACTION
    {
        ACT_BUTTON_UP,
        ACT_BUTTON_DOWN,
        ACT_MOVE,
    };
    enum BUTTON
    {
        BU_LEFT     = 0x01,
        BU_RIGHT    = 0x02,
        BU_MIDDLE   = 0x10,
    };
    enum CONTROL
    {
        CT_CONTROL  = 0x08,
        CT_SHIFT    = 0x04,
    };

    short action;
    short button;
    short control;
    short x;
    short y;
};




class KeyboardState
{
private:
    unsigned char   m_keystate[256];

public:
    enum KEY
    {
        KEY_ESCAPE = VK_ESCAPE,
    };

    KeyboardState() { memset(this, 0, sizeof(*this)); }
    unsigned char* getRawKeyState() { return m_keystate; }
    bool isKeyPressed(int v) const { return (m_keystate[v] & 0x80)!=0; }

};

class MouseState
{
public:
    enum BUTTON
    {
        BU_LEFT     = 0x01,
        BU_RIGHT    = 0x02,
        BU_MIDDLE   = 0x04,
    };

private:
    int m_button;
    int m_x;
    int m_y;

public:
    MouseState() { memset(this, 0, sizeof(*this)); }

    int getButtonState() const { return m_button; }
    int getX() const { return m_x; }
    int getY() const { return m_y; }

    void setButtonState(int v) { m_button=v; }
    void setX(int v) { m_x=v; }
    void setY(int v) { m_y=v; }
};

class JoyState
{
private:
    int m_x;
    int m_y;
    int m_z;
    int m_r;
    int m_u;
    int m_v;
    int m_pov;
    int m_buttons;

public:
    JoyState() { memset(this, 0, sizeof(*this)); }
    int getX() const { return m_x; } // -32768 Å` 32767
    int getY() const { return m_y; } // -32768 Å` 32767
    int getZ() const { return m_z; } // -32768 Å` 32767
    int getR() const { return m_r; } // -32768 Å` 32767
    int getU() const { return m_u; } // -32768 Å` 32767
    int getV() const { return m_v; } // -32768 Å` 32767
    int getRoV() const { return m_pov; } // 0 Å` 35900, degree * 100
    int getButtons() const { return m_buttons; }
    bool isButtonPressed(int i) const { return (m_buttons & (1<<i))!=0; }

    void setValue(const JOYINFOEX& v)
    {
        m_x = v.dwXpos; m_x -= 32768;
        m_y = v.dwYpos; m_y -= 32768;
        m_z = v.dwZpos; m_z -= 32768;
        m_r = v.dwRpos; m_r -= 32768;
        m_u = v.dwUpos; m_u -= 32768;
        m_v = v.dwVpos; m_v -= 32768;
        m_pov = v.dwPOV;
        m_buttons = v.dwButtons;
    }
};



class Application
{
private:
    static const int MAX_JOYSTICK_NUM = 4;

    HWND        m_hwnd;
    DEVMODE     m_devmode;
    bool        m_fullscreen;

    KeyboardState   m_keyboard_state;
    MouseState      m_mouse_state;
    JoyState        m_joy_state[MAX_JOYSTICK_NUM];

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

    size_t m_width, m_height;

public:
    static Application* getInstance();

    Application();
    virtual ~Application();

    virtual bool initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen=false);
    virtual void finalize();

    virtual bool initializeDraw();
    virtual void finalizeDraw();

    virtual void mainLoop()=0;
    virtual int handleWindowMessage(const WindowMessage& wm)=0;

    virtual void updateInput();
    virtual void translateMessage();

#ifdef IST_OPENCL
    cl::Context* getCLContext() { return m_cl_context; }
    cl::CommandQueue* getCLCommandQueue() { return m_cl_queue; }
#endif // IST_OPENCL

    size_t getWindowWidth() const   { return m_width; }
    size_t getWindowHeight() const  { return m_height; }
    const KeyboardState& getKeyboardState() const   { return m_keyboard_state; }
    const MouseState& getMouseState() const         { return m_mouse_state; }
    const JoyState& getJoyState(int i=0) const      { return m_joy_state[i]; }
#if defined(IST_OPENGL) && defined(WIN32)
    HDC     getHDC() const { return m_hdc; }
    HGLRC   getHGLRC() const { return m_hglrc; }
#endif // IST_OPENGL WIN32
};

} // namespace ist

#define istGetAplication()  ist::Application::getInstance()
#endif // __ist_Application__
