#ifndef atm_UI_Title_h
#define atm_UI_Title_h
#include "ist/iui.h"

namespace atm {

class TitleWindow;
class StartWindow;
class RecordWindow;
class ConfigWindow;
class LogWindow;

class RootWindow : public iui::RootWindow
{
typedef iui::RootWindow super;
public:
    RootWindow();
    virtual void update(iui::Float dt=0.0f);

private:
    TitleWindow     *m_title;
    LogWindow       *m_log;
};


class TitleWindow : public iui::Panel
{
public:
    TitleWindow();
    virtual void draw();

private:
    void onStart(Widget *);
    void onRecord(Widget *);
    void onConfig(Widget *);
    void onExit(Widget *);
    void hideAll();

    StartWindow     *m_start;
    RecordWindow    *m_record;
    ConfigWindow    *m_config;

    iui::ToggleButton *m_buttons[4];
};

class StartWindow : public iui::Panel
{
public:
    StartWindow();
private:
    void onCampaign(Widget *);
    void onHorde(Widget *);
    void onQuickJoin(Widget *);
};

class RecordWindow : public iui::Panel
{
public:
    RecordWindow();
private:
    void onSelect(Widget *);
    void onStart(Widget *);
    iui::String m_selection;
};


class ConfigWindow : public iui::Panel
{
public:
    ConfigWindow();
private:
    void onName(Widget *);
    void onFullscreen(Widget *);
    void onResolution(Widget *);
    void onBGMVolume(Widget *);
    void onBGMOnOff(Widget *);
    void onSEVolume(Widget *);
    void onSEOnOff(Widget *);
};

class LogWindow : public iui::Panel
{
public:
    LogWindow();
private:
    void onTextEnter(Widget *);
};
} // namespace atm
#endif // atm_UI_Title_h
