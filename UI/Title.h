#ifndef atomic_UI_Title_h
#define atomic_UI_Title_h
#include "ist/iui.h"

namespace atomic {

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
    ConfigWindow    *m_config;
    LogWindow       *m_log;
};
iui::RootWindow* CreateRootWindow()
{
    return istNew(RootWindow)();
}


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
};


class ConfigWindow : public iui::Panel
{
public:
    ConfigWindow();
private:
    void onName(Widget *);
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
} // namespace atomic
#endif // atomic_UI_Title_h
