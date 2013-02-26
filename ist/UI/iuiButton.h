#ifndef iui_Button_h
#define iui_Button_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class ButtonStyle : public Style
{
public:
    virtual void draw();
};

class Button : public Widget
{
public:
    Button();
    Button(const wchar_t *text, const WidgetCallback &on_press);

private:
    WidgetCallback m_on_press;
    bool m_clicking;
};


class ToggleButtonStyle : public Style
{
public:
    virtual void draw();
};

class ToggleButton : public Widget
{
public:
    ToggleButton();
    ToggleButton(const wchar_t *text, const WidgetCallback &on_toggle);
    bool isPressed() const;

private:
    WidgetCallback m_on_toggle;
    bool m_pressed;
    bool m_clicking;
};


class CheckboxButtonStyle : public Style
{
public:
    virtual void draw();
};

class Checkbox : public Widget
{
public:
    Checkbox();
    Checkbox(const wchar_t *text, const WidgetCallback &on_toggle);
    bool isChecked() const;

private:
    WidgetCallback m_on_toggle;
    bool m_checked;
    bool m_clicking;
};


class SliderStyle : public Style
{
public:
    virtual void draw();
};

class Slider : public Widget
{
public:
    Slider();
    Slider(const wchar_t *text, const WidgetCallback &on_change);
    Float getValue() const;

private:
    WidgetCallback m_on_change;
    Range m_range;
    Float m_position;
    bool m_dragging;
};


class EditboxStyle : public Style
{
public:
    virtual void draw();
};

class Editbox : public Widget
{
public:
    Editbox();
    Editbox(const wchar_t *text, const WidgetCallback &on_change);
    bool isReadOnly() const;
    int32 getCursor() const;

private:
    WidgetCallback m_on_change;
    bool m_readonly;
    int32 m_cursor;
};

class EditboxMultilineStyle : public Style
{
public:
    virtual void draw();
};

class EditboxMultiline : public Widget
{
public:
    EditboxMultiline();
    EditboxMultiline(const wchar_t *text, const WidgetCallback &on_change);
    bool isReadOnly() const;
    ivec2 getCursor() const;

private:
    WidgetCallback m_on_change;
    bool m_readonly;
    ivec2 m_cursor;
};

} // namespace iui
#endif // iui_Button_h
