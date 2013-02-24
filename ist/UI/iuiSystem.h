#ifndef ist_UI_iuiSystem_h
#define ist_UI_iuiSystem_h
#include "iuiCommon.h"
namespace ist {
namespace iui {

class UIRenderer;

class istInterModule UISystem : public SharedObject
{
public:
    typedef Application::WMHandler WMHandler;

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
} // namespace ist

#define iuiInitialize() ist::iui::UISystem::initializeInstance()
#define iuiFinalize()   ist::iui::UISystem::finalizeInstance()
#define iuiGetSystem()  ist::iui::UISystem::getInstance()

#endif // ist_UI_iuiSystem_h
