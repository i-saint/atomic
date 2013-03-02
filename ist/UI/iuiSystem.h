#ifndef iui_System_h
#define iui_System_h
#include "iuiCommon.h"
#include "iuiEvent.h"
namespace iui {


class iuiInterModule UISystem : public ist::SharedObject
{
public:
    typedef ist::Application::WMHandler WMHandler;

    static void initializeInstance();
    static void finalizeInstance();
    static UISystem* getInstance();

    void update(Float dt=0.0f);
    void draw();

    UIRenderer* getRenderer() const;
    Widget*     getRootWindow() const;
    Widget*     getFocus() const;
    const Rect& getScreen() const;

    void setScreen(float32 width, float32 height);
    void setScreen(float32 left, float32 right, float32 bottom, float32 top);
    void setFocus(Widget *v);

private:
    UISystem();
    ~UISystem();
    bool handleWindowMessage(const ist::WM_Base &wm);
    bool handleWindowMessageR(Widget *widget, const WM_Base &wm);
    void updateR(Widget *widget, Float dt);
    void drawR(Widget *widget);

    static UISystem *s_inst;
    istMemberPtrDecl(Members) m;
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
#define iuiGetRootWindow()          iuiGetSystem()->getRootWindow()
#define iuiGetRenderer()            iuiGetSystem()->getRenderer()


#endif // iui_System_h
