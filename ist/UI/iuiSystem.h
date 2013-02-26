#ifndef iui_System_h
#define iui_System_h
#include "iuiCommon.h"
namespace iui {


class istInterModule UISystem : public ist::SharedObject
{
public:
    typedef ist::Application::WMHandler WMHandler;

    static void initializeInstance();
    static void finalizeInstance();
    static UISystem* getInstance();

    void update(Float dt);
    void draw();

    UIRenderer* getRenderer() const;
    Widget* getRootWidgets() const;
    Widget* getFocus() const;

    void setFocus(Widget *v);

private:
    UISystem();
    ~UISystem();
    bool handleWindowMessage(const WM_Base &wm);

    static UISystem *s_inst;
    struct Members;
    deep_copy_ptr<Members> m;
};

} // namespace iui

#define iuiInitialize() iui::UISystem::initializeInstance()
#define iuiFinalize()   iui::UISystem::finalizeInstance()
#define iuiGetSystem()  iui::UISystem::getInstance()

#endif // iui_System_h
