#ifndef ist_Application_h
#define ist_Application_h

#include "WindowMessage.h"
#include "Input.h"


namespace ist {


class istInterModule Application
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
    typedef std::function<bool (const WM_Base&)> WMHandler;

public:
    static Application* getInstance();

    Application();
    virtual ~Application();

    virtual bool initialize(ivec2 wpos, ivec2 wsize, const wchar_t *title, bool fullscreen=false);
    virtual void finalize();

    virtual void mainLoop()=0;

    virtual void translateMessage();

    int showMessageDialog(const char* message, const char* caption, int dlgtype=DLG_OK);
    int showMessageDialog(const wchar_t* message, const wchar_t* caption, int dlgtype=DLG_OK);

    bool isFullscreen() const;
    const uvec2& getWindowSize() const;

    DisplaySetting getCurrentDisplaySetting() const;
    void getAvalableDisplaySettings(DisplaySetting*& settings, int& num_settings) const;

    void addMessageHandler(WMHandler *wmh);
    void eraseMessageHandler(WMHandler *wmh);

#ifdef ist_env_Windows
    HWND getWindowHandle() const;
#endif // ist_env_Windows


private:
    static const int MAX_JOYSTICK_NUM = 4;

    istMemberPtrDecl(Members) m;

#ifdef ist_env_Windows
    bool _handleWindowMessage(const WM_Base& wm);
    friend LRESULT CALLBACK istWndProc(HWND hwnd , UINT message , WPARAM wParam , LPARAM lParam);
#endif // ist_env_Windows
};

} // namespace ist


#define istGetAplication()  ist::Application::getInstance()
#define istShowMessageDialog(mes, cap, dtype) istGetAplication()->showMessageDialog(mes, cap, dtype)


#endif // ist_Application_h
