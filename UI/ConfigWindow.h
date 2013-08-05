#ifndef atm_UI_ConfigMenu_h
#define atm_UI_ConfigMenu_h
#include "ist/iui.h"

namespace atm {


class EditboxWindow : public iui::Panel
{
public:
    EditboxWindow(Widget *parent, const wchar_t *value, const iui::Rect &rect, iui::WidgetCallback on_value)
    {
        setParent(parent);
        setPosition(rect.getPosition());
        setSize(rect.getSize());
        m_ed = iuiNew(iui::Editbox)(this, value, iui::Rect(iui::Position(), iui::Size(100, 20)), on_value);
    }

    iui::Editbox* getEdit() const { return m_ed; }
    bool onCancel(const iui::WM_Widget &wm) { return getParent()->onCancel(wm); }

private:
    iui::Editbox *m_ed;
};

class ListWindow : public iui::Panel
{
public:
    ListWindow(Widget *parent, const iui::Rect &rect, iui::WidgetCallback on_value)
    {
        setParent(parent);
        setPosition(rect.getPosition());
        setSize(rect.getSize());
        m_li = iuiNew(iui::List)(this, iui::Rect(iui::Position(), iui::Size(200, 150)), on_value);
    }

    iui::List* getList() const { return m_li; }
    bool onCancel(const iui::WM_Widget &wm) { return getParent()->onCancel(wm); }

private:
    iui::List *m_li;
};

template<class IntegerOrFloat>
class NumericWindow : public iui::Panel
{
public:
    typedef IntegerOrFloat value_t;
    NumericWindow(Widget *parent, const iui::Rect &rect=iui::Rect(), iui::WidgetCallback on_value=iui::WidgetCallback())
        : m_value(), m_step(), m_min(), m_max()
    {
        setParent(parent);
        setPosition(rect.getPosition());
        setSize(rect.getSize());
        setMin(min);
        setMax(max);
        setStep(step);
        setValue(value);
        const iui::Size &size = rect.getSize();
        m_bu_up   = iuiNew(iui::Button)(this, L"△", iui::Rect(iui::Position( 25.0f,  0.0f), iui::Size( 50.0f, 20.0f)), std::bind(&NumericWindow::onButton, this, _1));
        m_lb_value= iuiNew(iui::Label )(this, L"",   iui::Rect(iui::Position(  0.0f, 25.0f), iui::Size(100.0f, 20.0f)));
        m_bu_down = iuiNew(iui::Button)(this, L"▽", iui::Rect(iui::Position( 25.0f, 50.0f), iui::Size( 50.0f, 20.0f)), std::bind(&NumericWindow::onButton, this, _1));
        m_on_value= on_value;
    }

    value_t getMin() const     { return m_min; }
    value_t getMax() const     { return m_max; }
    value_t getStep() const    { return m_step; }
    value_t getValue() const   { return m_value; }
    void    setMin(value_t v)  { m_min=v; }
    void    setMax(value_t v)  { m_max=v; }
    void    setStep(value_t v) { m_step=v; }

    void setValue(value_t v)
    {
        m_value = v;
        std::string str;
        ist::Stringnize(m_value, str);
        m_lb_value->setText(ist::L(str));
        callIfValid(m_on_value);
    }

    void setParams(value_t v, value_t step, value_t min, value_t max)
    {
        setMin(min); setMax(max); setStep(step); setValue(v);
    }

    bool onCancel(const iui::WM_Widget &wm) { return getParent()->onCancel(wm); }

private:
    void onButton(iui::Widget *w)
    {
        value_t v = getValue();
        if(w==m_bu_up)   { v+=m_step; }
        if(w==m_bu_down) { v-=m_step; }
        v = ist::clamp<value_t>(v, m_min, m_max);
        setValue(v);
    }

    iui::WidgetCallback m_on_value;
    iui::Button *m_bu_up;
    iui::Button *m_bu_down;
    iui::Label  *m_lb_value;
    value_t m_value;
    value_t m_step;
    value_t m_min;
    value_t m_max;
};


class ConfigWindow : public iui::Panel
{
typedef iui::Panel super;
public:
    ConfigWindow();
    void setVisibility(bool v, bool e=true) override;
    bool onCancel(const iui::WM_Widget &wm) override;
    void sync();
    void hideAll();

private:
    void onNameW(Widget *);
    void onNameV(Widget *);
    void onResolutionW(Widget *);
    void onResolutionV(Widget *);
    void onPortW(Widget *);
    void onPortV(Widget *);
    void onRenderW(Widget *);
    void onRenderV(Widget *);

    void onBGMW(Widget *);
    void onBGMV(Widget *);
    void onSEW(Widget *);
    void onSEV(Widget *);


    EditboxWindow           *m_vw_name;
    ListWindow              *m_vw_reso;
    EditboxWindow           *m_vw_http;
    ListWindow              *m_vw_glevel;

    iui::ToggleButton *m_bu_name;
    iui::ToggleButton *m_bu_reso;
    iui::ToggleButton *m_bu_http;
    iui::ToggleButton *m_bu_glevel;

    iui::Label *m_lb_desc;

    ist::vector<iui::Panel*>        m_windows;
    ist::vector<iui::ToggleButton*> m_buttons;
};

} // namespace atm
#endif // atm_UI_ConfigMenu_h
