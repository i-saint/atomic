#include "iuiPCH.h"
#include "iuiButton.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"
namespace iui {

void ButtonStyle::draw()
{
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}

struct Button::Members
{
    WidgetCallback on_press;
    bool pressing;

    Members() : pressing(false)
    {
    }
};

Button::Button( const wchar_t *text, const WidgetCallback &on_press, Style *style )
{
    m->on_press = on_press;
    setText(text);
    setStyle(style);
}
bool Button::isPressing() const             { return m->pressing;  }
Style* Button::createDefaultStyle() const   { return istNew(ButtonStyle)(); }



void ToggleButtonStyle::draw()
{
    // todo
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}

struct ToggleButton::Members
{
    WidgetCallback on_toggle;
    bool pressed;
    bool pressing;

    Members() : pressed(false), pressing(false)
    {}
};

ToggleButton::ToggleButton( const wchar_t *text, const WidgetCallback &on_toggle, Style *style )
{
    m->on_toggle = on_toggle;
    setText(text);
    setStyle(style);
}

bool ToggleButton::isPressed() const    { return m->pressed; }
bool ToggleButton::isPressing() const   { return m->pressing; }
Style* ToggleButton::createDefaultStyle() const { return istNew(ToggleButtonStyle)(); }



void CheckboxStyle::draw()
{
    // todo
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}

struct Checkbox::Members
{
    WidgetCallback on_toggle;
    bool checked;
    bool pressing;

    Members() : checked(false), pressing(false)
    {}
};

Checkbox::Checkbox( const wchar_t *text, const WidgetCallback &on_toggle, Style *style )
{
    m->on_toggle = on_toggle;
    setText(text);
    setStyle(style);
}

bool Checkbox::isChecked() const    { return m->checked; }
bool Checkbox::isPressing() const   { return m->pressing; }
Style* Checkbox::createDefaultStyle() const { return istNew(CheckboxStyle)(); }


} // namespace iui
