#ifndef atm_UI_ConfigMenu_h
#define atm_UI_ConfigMenu_h
#include "ist/iui.h"

namespace atm {

class ConfigWindow : public iui::Panel
{
public:
    ConfigWindow();
    void sync();

private:
    void onName(Widget *);
    void onFullscreen(Widget *);
    void onResolution(Widget *);
    void onBGMVolume(Widget *);
    void onBGMOnOff(Widget *);
    void onSEVolume(Widget *);
    void onSEOnOff(Widget *);
    bool onCancel(const iui::WM_Widget &wm) override;
};

} // namespace atm
#endif // atm_UI_ConfigMenu_h
