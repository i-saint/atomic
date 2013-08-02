#ifndef atm_UI_LogWindow_h
#define atm_UI_LogWindow_h
#include "ist/iui.h"

namespace atm {

class LogWindow : public iui::Panel
{
typedef iui::Panel super;
friend class RootWindow;
private:
    LogWindow();
    void drawCallback();
    void onTextEnter(Widget *);
};
} // namespace atm

#endif // atm_UI_LogWindow_h
