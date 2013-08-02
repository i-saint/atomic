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
    void setVisibility(bool v) override;
    void unselectAll();
    bool onCancel(const iui::WM_Widget &wm) override;

private:
    void onResume(Widget *);
    void onConfig(Widget *);
    void onTitle(Widget *);
    void onExit(Widget *);

    ConfigWindow       *m_config;
    iui::ToggleButton  *m_buttons[4];
};

} // namespace atm
#endif // atm_UI_PauseMenu_h
