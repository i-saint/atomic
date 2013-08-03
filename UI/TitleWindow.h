#ifndef atm_UI_TitleMenu_h
#define atm_UI_TitleMenu_h
#include "ist/iui.h"

namespace atm {

class StartWindow;
class RecordWindow;
class ConfigWindow;

class TitleWindow : public iui::Panel
{
typedef iui::Panel super;
friend class RootWindow;
private:
    TitleWindow();
public:
    void update(iui::Float dt) override;
    void draw() override;
    void drawCallback();

    void setVisibility(bool v, bool e) override;
    void unselectAll();
    bool onCancel(const iui::WM_Widget &wm) override;

private:
    void onStart(Widget *);
    void onRecord(Widget *);
    void onConfig(Widget *);
    void onExit(Widget *);

    StartWindow        *m_start;
    RecordWindow       *m_record;
    ConfigWindow       *m_config;
    iui::ToggleButton  *m_buttons[4];
    float32             m_time;
};

} // namespace atm
#endif // atm_UI_TitleMenu_h
