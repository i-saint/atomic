#include "iuiPCH.h"
#include "iuiButton.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"
#include "iuiUtilities.h"
namespace iui {


ButtonStyle::ButtonStyle()
{
    setTextHAlign(TA_HCenter);
    setTextHSpacing(1.0f);
}

void ButtonStyle::draw()
{
    Button *w = static_cast<Button*>(getWidget());
    Rect rect(Position(), w->getSize());
    TextPosition tpos(rect, getTextHAlign(), getTextVAlign(), getTextHSpacing(), getTextVSpacing());
    Color bg = getBGColor();
    if(w->isPressing()) {
        bg += vec4(0.4f, 0.4f, 0.4f, 0.0f);
    }
    else if(w->isHovered()) {
        bg += vec4(0.2f, 0.2f, 0.2f, 0.0f);
    }
    iuiGetRenderer()->drawRect(rect, bg);
    iuiGetRenderer()->drawOutlineRect(rect, getBorderColor());
    iuiGetRenderer()->drawFont(tpos, getFontColor(), w->getText().c_str(), w->getText().size());
}
iuiImplDefaultStyle(Button);

struct Button::Members
{
    WidgetCallback on_press;
    bool pressing;
    bool hovered;

    Members() : pressing(false), hovered(false)
    {
    }
};
istMemberPtrImpl(Button,Members);

bool Button::isPressing() const { return m->pressing;  }
bool Button::isHovered() const  { return m->hovered; }

Button::Button( Widget *parent, const wchar_t *text, const Rect &rect, WidgetCallback on_press )
{
    setParent(parent);
    setText(text);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    m->on_press = on_press;
}

void Button::update(Float dt)
{
    HandleMouseHover(this, m->hovered);
    super::update(dt);
}

bool Button::handleEvent( const WM_Base &wm )
{
    switch(MouseHitWidget(this, wm)) {
    case WH_HitMouseLeftDown:
        setFocus(true);
        m->pressing = true;
        return true;
    case WH_HitMouseLeftUp:
        callIfValid(m->on_press);
        m->pressing = false;
        return true;
    case WH_MissMouseLeftUp:
        m->pressing = false;
        break;
    }
    return super::handleEvent(wm);
}



void ToggleButtonStyle::draw()
{
    ToggleButton *w = static_cast<ToggleButton*>(getWidget());
    Rect rect(Position(), w->getSize());
    TextPosition tpos(rect, getTextHAlign(), getTextVAlign(), getTextHSpacing(), getTextVSpacing());
    Color bg = getBGColor();
    if(w->isPressing()) {
        bg += vec4(0.4f, 0.4f, 0.4f, 0.0f);
    }
    else if(w->isHovered()) {
        bg += vec4(0.2f, 0.2f, 0.2f, 0.0f);
    }
    iuiGetRenderer()->drawRect(rect, bg);
    iuiGetRenderer()->drawOutlineRect(rect, getBorderColor());
    iuiGetRenderer()->drawFont(tpos, getFontColor(), w->getText().c_str(), w->getText().size());
}
iuiImplDefaultStyle(ToggleButton);

struct ToggleButton::Members
{
    WidgetCallback on_toggle;
    bool pressed;
    bool pressing;
    bool hovered;

    Members() : pressed(false), pressing(false), hovered(false)
    {}
};
istMemberPtrImpl(ToggleButton,Members);

bool ToggleButton::isPressed() const    { return m->pressed; }
bool ToggleButton::isPressing() const   { return m->pressing; }
bool ToggleButton::isHovered() const    { return m->hovered; }

ToggleButton::ToggleButton( const wchar_t *text, WidgetCallback on_toggle )
{
    m->on_toggle = on_toggle;
    setText(text);
}

void ToggleButton::update( Float dt )
{
    HandleMouseHover(this, m->hovered);
    super::update(dt);
}

bool ToggleButton::handleEvent( const WM_Base &wm )
{
    switch(MouseHitWidget(this, wm)) {
    case WH_HitMouseLeftDown:
        setFocus(true);
        m->pressing = true;
        return true;
    case WH_HitMouseLeftUp:
        if(m->pressing) {
            m->pressed = !m->pressed;
            callIfValid(m->on_toggle);
            m->pressing = false;
        }
        return true;
    case WH_MissMouseLeftUp:
        m->pressing = false;
        break;
    }
    return super::handleEvent(wm);
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

Checkbox::Checkbox( const wchar_t *text, WidgetCallback on_toggle )
{
    m->on_toggle = on_toggle;
    setText(text);
}

bool Checkbox::handleEvent( const WM_Base &wm )
{
    switch(MouseHitWidget(this, wm)) {
    case WH_HitMouseLeftDown:
        m->pressing = true;
        return true;
    case WH_HitMouseLeftUp:
        if(m->pressing) {
            m->checked = !m->checked;
            callIfValid(m->on_toggle);
            m->pressing = false;
        }
        return true;
    }
    return super::handleEvent(wm);
}


} // namespace iui
