#ifndef iui_Button_h
#define iui_Button_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiAPI ButtonStyle : public Style
{
public:
    ButtonStyle();
    virtual void draw();
};

class iuiAPI Button : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Button);

    Button(Widget *parent, const wchar_t *text=L"", const Rect &rect=Rect(), WidgetCallback on_press=WidgetCallback());
    void update(Float dt) override;
    bool isPressing() const;
    bool isHovered() const;

    bool handleEvent(const WM_Base &wm) override;
    bool onOK(const WM_Widget &em) override;
    virtual void push();

private:
    WidgetCallback m_on_press;
    bool m_pressing;
    bool m_hovered;
};


class iuiAPI ToggleButtonStyle : public Style
{
public:
    ToggleButtonStyle();
    virtual void draw();
};

class iuiAPI ToggleButton : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(ToggleButton)
    ToggleButton(Widget *parent, const wchar_t *text=L"", const Rect &rect=Rect(), WidgetCallback on_toggle=WidgetCallback());
    void update(Float dt) override;
    bool isPressed() const;
    bool isPressing() const;
    bool isHovered() const;
    void setPressed(bool v, bool fire_event=true);

    bool handleEvent(const WM_Base &wm) override;
    bool onOK(const WM_Widget &em) override;
    virtual void toggle();

private:
    WidgetCallback m_on_toggle;
    bool m_pressed;
    bool m_pressing;
    bool m_hovered;
};


class iuiAPI CheckboxStyle : public Style
{
public:
    virtual void draw();
};

class iuiAPI Checkbox : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(Checkbox)
    Checkbox(const wchar_t *text=L"", WidgetCallback on_toggle=WidgetCallback());
    bool isChecked() const;
    bool isPressing() const;

    bool handleEvent(const WM_Base &wm) override;
    bool onOK(const WM_Widget &em) override;
    virtual void toggle();

private:
    WidgetCallback m_on_toggle;
    bool m_checked;
    bool m_pressing;
};


} // namespace iui
#endif // iui_Button_h
