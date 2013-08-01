#ifndef atm_UI_Title_h
#define atm_UI_Title_h
#include "ist/iui.h"

namespace atm {

class UICursor;
class TitleWindow;
class StartWindow;
class RecordWindow;
class ConfigWindow;
class LogWindow;

class UICursor
{
public:
    struct State {
        iui::Widget *widget;
        uint32 index;
        State(iui::Widget *w=nullptr, uint32 i=0) : widget(w), index(i) {}
    };

    UICursor();
    void update(iui::Float dt);
    void draw();

    void pushStack(iui::Widget *v);
    void popStack();
    void clearStack();
    void setTarget(iui::Widget *v);

    void moveNext();
    void movePrev();
    void enter();
    void cancel();

private:
    ist::vector<State> m_stack;
    iui::Position m_pos;
    iui::Size m_size;
};


class RootWindow : public iui::RootWindow
{
typedef iui::RootWindow super;
public:
    RootWindow();
    ~RootWindow();
    void update(iui::Float dt=0.0f) override;
    void draw() override;

    UICursor* getCursor() { return m_cursor; }

private:
    TitleWindow     *m_title;
    LogWindow       *m_log;
    UICursor        *m_cursor;
};
UICursor* atmGetUICursor();

class TitleWindow : public iui::Panel
{
typedef iui::Panel super;
public:
    TitleWindow();
    void draw() override;
    void setVisibility(bool v) override;

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
    void onEdit(Widget *);
};

class RecordWindow : public iui::Panel
{
public:
    RecordWindow();
    void refresh();
private:
    void onSelect(Widget *);
    void onStart(Widget *);

    iui::List   *m_li_files;
    iui::Button *m_bu_start;
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
