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


bool Button::isPressing() const { return m_pressing;  }
bool Button::isHovered() const  { return m_hovered; }

Button::Button( Widget *parent, const wchar_t *text, const Rect &rect, WidgetCallback on_press )
    : m_pressing(false), m_hovered(false)
{
    setParent(parent);
    setText(text);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    m_on_press = on_press;
    setupDefaultParams();
}

void Button::update(Float dt)
{
    HandleMouseHover(this, m_hovered);
    super::update(dt);
}

bool Button::handleEvent( const WM_Base &wm )
{
    switch(MouseHit(this, wm)) {
    case WH_HitMouseLeftDown:
        setFocus(true);
        m_pressing = true;
        return true;
    case WH_HitMouseLeftUp:
        push();
        m_pressing = false;
        return true;
    case WH_MissMouseLeftUp:
        m_pressing = false;
        break;
    }
    return super::handleEvent(wm);
}

bool Button::onOK(const WM_Widget &em)
{
    setFocus(true);
    push();
    return true;
}

void Button::push()
{
    callIfValid(m_on_press);
}



ToggleButtonStyle::ToggleButtonStyle()
{
    setTextHAlign(TA_HCenter);
    setTextHSpacing(1.0f);
}

void ToggleButtonStyle::draw()
{
    ToggleButton *w = static_cast<ToggleButton*>(getWidget());
    Rect rect(Position(), w->getSize());
    TextPosition tpos(rect, getTextHAlign(), getTextVAlign(), getTextHSpacing(), getTextVSpacing());
    Color bg = getBGColor();
    if(w->isPressing() || w->isPressed()) {
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

bool ToggleButton::isPressed() const    { return m_pressed; }
bool ToggleButton::isPressing() const   { return m_pressing; }
bool ToggleButton::isHovered() const    { return m_hovered; }
void ToggleButton::setPressed(bool v, bool fire_event)
{
    bool prev = m_pressed;
    m_pressed = v;
    if(fire_event && prev!=v) {
        callIfValid(m_on_toggle);
    }
}

ToggleButton::ToggleButton( Widget *parent, const wchar_t *text, const Rect &rect, WidgetCallback on_toggle )
    : m_pressed(false), m_pressing(false), m_hovered(false)
{
    setParent(parent);
    setText(text);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    m_on_toggle = on_toggle;
    setupDefaultParams();
}

void ToggleButton::update( Float dt )
{
    HandleMouseHover(this, m_hovered);
    super::update(dt);
}

bool ToggleButton::handleEvent( const WM_Base &wm )
{
    switch(MouseHit(this, wm)) {
    case WH_HitMouseLeftDown:
        setFocus(true);
        m_pressing = true;
        return true;
    case WH_HitMouseLeftUp:
        if(m_pressing) {
            toggle();
            m_pressing = false;
        }
        return true;
    case WH_MissMouseLeftUp:
        m_pressing = false;
        break;
    }
    return super::handleEvent(wm);
}

bool ToggleButton::onOK(const WM_Widget &em)
{
    setFocus(true);
    toggle();
    return true;
}

void ToggleButton::toggle()
{
    setPressed(!m_pressed);
}



void CheckboxStyle::draw()
{
    // todo
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}
iuiImplDefaultStyle(Checkbox);


bool Checkbox::isChecked() const    { return m_checked; }
bool Checkbox::isPressing() const   { return m_pressing; }

Checkbox::Checkbox( const wchar_t *text, WidgetCallback on_toggle )
    : m_checked(false), m_pressing(false)
{
    m_on_toggle = on_toggle;
    setText(text);
    setupDefaultParams();
}

bool Checkbox::handleEvent( const WM_Base &wm )
{
    switch(MouseHit(this, wm)) {
    case WH_HitMouseLeftDown:
        m_pressing = true;
        return true;
    case WH_HitMouseLeftUp:
        if(m_pressing) {
            toggle();
            m_pressing = false;
        }
        return true;
    }
    return super::handleEvent(wm);
}

bool Checkbox::onOK(const WM_Widget &em)
{
    setFocus(true);
    toggle();
    return true;
}

void Checkbox::toggle()
{
    m_checked = !m_checked;
    callIfValid(m_on_toggle);
}


} // namespace iui
