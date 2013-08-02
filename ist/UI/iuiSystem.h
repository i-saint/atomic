#ifndef iui_System_h
#define iui_System_h
#include "iuiCommon.h"
#include "iuiEvent.h"
namespace iui {


class iuiAPI UISystem
{
friend class Widget;
public:
    typedef ist::Application::WMHandler WMHandler;

    static void initializeInstance();
    static void finalizeInstance();
    static UISystem* getInstance();

    void update(Float dt=1.0f);
    void draw();

    UIRenderer*     getRenderer() const;
    RootWindow*     getRootWindow() const;
    Widget*         getFocus() const;
    const Position& getMousePos() const;
    const Rect&     getScreen() const;

    void setRootWindow(RootWindow *root);
    void setScreen(float32 width, float32 height);
    void sendMessage(const WM_Base &wm);

private:
    // Widget から呼ばれる系
    void setFocus(Widget *v);
    void notifyNewWidget(Widget *w);

private:
    UISystem();
    ~UISystem();
    bool handleWindowMessage(const ist::WM_Base &wm);
    bool handleWindowMessageR(Widget *widget, const WM_Base &wm);
    void updateR(Widget *widget, Float dt);
    void drawR(Widget *widget);

    typedef UISystem::WMHandler WMHandler;
    static UISystem *s_inst;
    UIRenderer *m_renderer;
    RootWindow *m_root;
    Widget     *m_focus;
    WidgetCont  m_new_widgets;
    WMHandler   m_wmhandler;
    Position    m_mouse_pos;
    Rect        m_screen;
};

} // namespace iui

#define iuiInitialize()             iui::UISystem::initializeInstance()
#define iuiFinalize()               iui::UISystem::finalizeInstance()
#define iuiInitializeRenderer(...)  iuiGetRenderer()->initialize(__VA_ARGS__)
#define iuiFinalizeRenderer()       iuiGetRenderer()->finalize()
#define iuiUpdate()                 iuiGetSystem()->update()
#define iuiDraw()                   iuiGetSystem()->draw()
#define iuiDrawFlush()              iuiGetRenderer()->flush()

#define iuiGetSystem()              iui::UISystem::getInstance()
#define iuiGetRenderer()            iuiGetSystem()->getRenderer()
#define iuiGetRootWindow()          iuiGetSystem()->getRootWindow()
#define iuiGetMousePos()            iuiGetSystem()->getMousePos()


#endif // iui_System_h
