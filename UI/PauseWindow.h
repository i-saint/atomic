#ifndef atm_UI_PauseMenu_h
#define atm_UI_PauseMenu_h
#include "ist/iui.h"

namespace atm {

class ConfigWindow;

class PauseWindow : public iui::Panel
{
typedef iui::Panel super;
friend class RootWindow;
private:
    PauseWindow();
public:
    void setVisibility(bool v, bool e=true) override;
    void unselectAll();
    bool onCancel(const iui::WM_Widget &wm) override;
    void drawCallback();

private:
    void onResume(Widget *);
    void onConfig(Widget *);
    void onTitle(Widget *);
    void onExit(Widget *);
    void onState(Widget *);

    ConfigWindow       *m_config;
    iui::ToggleButton  *m_buttons[5];
};

} // namespace atm
#endif // atm_UI_PauseMenu_h
