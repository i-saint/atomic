#ifndef atm_Engine_UI_RootWindow_h
#define atm_Engine_UI_RootWindow_h
#include "ist/iui.h"

namespace atm {

class UISelector;
class TitleWindow;
class PauseWindow;
class ConfigWindow;
class LogWindow;

iui::RootWindow*    atmCreateRootWindow();
iui::RootWindow*    atmGetRootWindow();
iui::Widget*        atmGetTitleWindow();
iui::Widget*        atmGetPauseWindow();
iui::Widget*        atmGetConfigWindow();
iui::Widget*        atmGetLogWindow();
UISelector*         atmGetUISelector();


class RootWindow : public iui::RootWindow
{
typedef iui::RootWindow super;
friend iui::RootWindow* atmCreateRootWindow();
private:
    RootWindow();
    ~RootWindow();
public:
    void update(iui::Float dt=1.0f) override;
    void draw() override;
    void drawCallback();

    TitleWindow*    getTitleWindow() const  { return m_title; }
    PauseWindow*    getPauseWindow() const  { return m_pause; }
    LogWindow*      getLogWindow() const    { return m_log; }
    UISelector*     getCursor() const       { return m_cursor; }

private:
    TitleWindow     *m_title;
    PauseWindow     *m_pause;
    LogWindow       *m_log;
    UISelector      *m_cursor;
};

} // namespace atm
#endif // atm_Engine_UI_RootWindow_h
