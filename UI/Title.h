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

iui::RootWindow*    atmCreateRootWindow();
TitleWindow*        atmGetTitleWindow();
UICursor* atmGetUICursor();

class UICursor
{
public:
    struct Selection {
        iui::Widget *widget;
        int32 index;
        Selection(iui::Widget *w=nullptr, int32 i=0) : widget(w), index(i) {}
    };
    typedef ist::vector<Selection> SelectionCont;

    UICursor();
    void update(iui::Float dt);
    void draw();

    void pushSelection(iui::Widget *v, int32 i=-1);
    void popSelection();
    void clearSelection();
    void setSelection(iui::Widget *v, int32 i=-1);
    SelectionCont& getSelection();

    void moveNext();
    void movePrev();
    void enter();
    void cancel();

private:
    SelectionCont m_selection;
    iui::Position m_pos;
    iui::Size m_size;
    iui::Float m_time;
};


class RootWindow : public iui::RootWindow
{
typedef iui::RootWindow super;
public:
    RootWindow();
    ~RootWindow();
    void update(iui::Float dt=1.0f) override;
    void draw() override;

    UICursor* getCursor() { return m_cursor; }

private:
    TitleWindow     *m_title;
    LogWindow       *m_log;
    UICursor        *m_cursor;
};

class TitleWindow : public iui::Panel
{
typedef iui::Panel super;
public:
    TitleWindow();
    void draw() override;
    void setVisibility(bool v) override;
    void unselectAll();
    bool onCancel(const iui::WM_Widget &wm) override;

private:
    void onStart(Widget *);
    void onRecord(Widget *);
    void onConfig(Widget *);
    void onExit(Widget *);

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
    bool onCancel(const iui::WM_Widget &wm) override;
};

class RecordWindow : public iui::Panel
{
public:
    RecordWindow();
    void refresh();
private:
    void onSelect(Widget *);
    void onStart(Widget *);
    bool onCancel(const iui::WM_Widget &wm) override;

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
    bool onCancel(const iui::WM_Widget &wm) override;
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
