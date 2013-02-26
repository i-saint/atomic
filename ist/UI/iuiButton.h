#ifndef iui_Button_h
#define iui_Button_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiInterModule ButtonStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Button : public Widget
{
public:
    Button(const wchar_t *text=L"", const WidgetCallback &on_press=WidgetCallback(), Style *style=NULL);
    bool isPressing() const;

protected:
    virtual Style* createDefaultStyle() const;
private:
    struct Members;
    deep_copy_ptr<Members> m;
};


class iuiInterModule ToggleButtonStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule ToggleButton : public Widget
{
public:
    ToggleButton(const wchar_t *text=L"", const WidgetCallback &on_toggle=WidgetCallback(), Style *style=NULL);
    bool isPressed() const;
    bool isPressing() const;

protected:
    virtual Style* createDefaultStyle() const;
private:
    struct Members;
    deep_copy_ptr<Members> m;
};


class iuiInterModule CheckboxStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Checkbox : public Widget
{
public:
    Checkbox(const wchar_t *text=L"", const WidgetCallback &on_toggle=WidgetCallback(), Style *style=NULL);
    bool isChecked() const;
    bool isPressing() const;

protected:
    virtual Style* createDefaultStyle() const;
private:
    struct Members;
    deep_copy_ptr<Members> m;
};


class iuiInterModule SliderStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule Slider : public Widget
{
public:
    Slider();
    Slider(const wchar_t *text, const WidgetCallback &on_change);
    Float getValue() const;

protected:
    virtual Style* createDefaultStyle() const;
private:
    WidgetCallback m_on_change;
    Range m_range;
    Float m_position;
    bool m_dragging;
};


} // namespace iui
#endif // iui_Button_h
