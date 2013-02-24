#ifndef ist_UI_iuiButton_h
#define ist_UI_iuiButton_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace ist {
namespace iui {


class Panel : public Widget
{
public:

private:
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

class CheckBox : public Widget
{
public:
    CheckBox();
    CheckBox(const wchar_t *text, const WidgetCallback &on_toggle);
    bool isChecked() const;

private:
    WidgetCallback m_on_toggle;
    bool m_checked;
    bool m_clicking;
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
    Float m_value;
    bool m_dragging;
};

class TextBox : public Widget
{
public:
    TextBox();
    TextBox(const wchar_t *text, const WidgetCallback &on_change);
    bool isReadOnly() const;
    int32 getCursor() const;

private:
    WidgetCallback m_on_change;
    bool m_readonly;
    int32 m_cursor;
};

class TextBoxMultiline : public Widget
{
public:
    TextBoxMultiline();
    TextBoxMultiline(const wchar_t *text, const WidgetCallback &on_change);
    bool isReadOnly() const;
    ivec2 getCursor() const;

private:
    WidgetCallback m_on_change;
    bool m_readonly;
    ivec2 m_cursor;
};

} // namespace iui
} // namespace ist
#endif // ist_UI_iuiButton_h
