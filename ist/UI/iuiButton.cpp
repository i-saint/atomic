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
iuiImplDefaultStyle(Button);

struct Button::Members
{
    WidgetCallback on_press;
    bool pressing;

    Members() : pressing(false)
    {
    }
};
istMemberPtrImpl(Button,Members);

bool Button::isPressing() const             { return m->pressing;  }

Button::Button( const wchar_t *text, const WidgetCallback &on_press )
{
    m->on_press = on_press;
    setText(text);
}



void ToggleButtonStyle::draw()
{
    // todo
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}
iuiImplDefaultStyle(ToggleButton);

struct ToggleButton::Members
{
    WidgetCallback on_toggle;
    bool pressed;
    bool pressing;

    Members() : pressed(false), pressing(false)
    {}
};
istMemberPtrImpl(ToggleButton,Members);

bool ToggleButton::isPressed() const    { return m->pressed; }
bool ToggleButton::isPressing() const   { return m->pressing; }

ToggleButton::ToggleButton( const wchar_t *text, const WidgetCallback &on_toggle )
{
    m->on_toggle = on_toggle;
    setText(text);
}



void CheckboxStyle::draw()
{
    // todo
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}
iuiImplDefaultStyle(Checkbox);

struct Checkbox::Members
{
    WidgetCallback on_toggle;
    bool checked;
    bool pressing;

    Members() : checked(false), pressing(false)
    {}
};
istMemberPtrImpl(Checkbox,Members);

bool Checkbox::isChecked() const    { return m->checked; }
bool Checkbox::isPressing() const   { return m->pressing; }

Checkbox::Checkbox( const wchar_t *text, const WidgetCallback &on_toggle )
{
    m->on_toggle = on_toggle;
    setText(text);
}


} // namespace iui
