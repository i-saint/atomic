#ifndef ist_UI_iuiSystem_h
#define ist_UI_iuiSystem_h
#include "iuiCommon.h"
namespace ist {
namespace iui {

class UIRenderer;

class istInterModule UISystem : public SharedObject
{
public:
    UISystem();
    ~UISystem();

    void update(Float dt);
    void draw();

    UIRenderer* getRenderer() const { return m_renderer; }
    Widget* getWidget() const { return m_widget; }

private:
    UIRenderer *m_renderer;
    Widget *m_widget;
};

} // namespace iui
} // namespace ist
#endif // ist_UI_iuiSystem_h
