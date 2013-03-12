#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"
#include "iuiEdit.h"
#include "iuiUtilities.h"
namespace iui {


void LabelStyle::draw()
{
    Editbox *w = static_cast<Editbox*>(getWidget());
    TextPosition tpos(Rect(w->getPositionAbs(), w->getSize()), getTextHAlign(), getTextVAlign());
    iuiGetRenderer()->drawFont(tpos, getFontColor(), w->getText().c_str(), w->getText().size());
}


Label::Label( Widget *parent, const wchar_t *text, const Rect &rect )
{
    setParent(parent);
    setText(text);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
}

bool Label::handleEvent( const WM_Base &wm )
{
    return super::handleEvent(wm);
}



void EditboxStyle::draw()
{
    Editbox *w = static_cast<Editbox*>(getWidget());
    Rect rect(w->getPositionAbs(), w->getSize());
    TextPosition tpos(Rect(w->getPositionAbs(), w->getSize()), getTextHAlign(), getTextVAlign());
    Color bg = getBGColor();
    if(w->isFocused()) {
        bg += vec4(0.4f, 0.4f, 0.4f, 0.0f);
    }
    else if(w->isHovered()) {
        bg += vec4(0.2f, 0.2f, 0.2f, 0.0f);
    }
    iuiGetRenderer()->drawRect(rect, bg);
    iuiGetRenderer()->drawOutlineRect(rect, getBorderColor());
    iuiGetRenderer()->drawFont(tpos, getFontColor(), w->getText().c_str(), w->getText().size());
}
iuiImplDefaultStyle(Editbox);

struct Editbox::Members
{
    bool readonly;
    bool hovered;
    int32 cursor;

    Members() : readonly(false), hovered(false), cursor(0)
    {
    }
};
istMemberPtrImpl(Editbox,Members);

Editbox::Editbox(Widget *parent, const wchar_t *text, const Rect &rect, const WidgetCallback &on_edit)
{
    setParent(parent);
    setText(text);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    setTextHandler(on_edit);
}

int32   Editbox::getCursor() const      { return m->cursor; }

bool    Editbox::isHovered() const      { return m->hovered; }
bool    Editbox::isReadOnly() const     { return m->readonly; }
void    Editbox::setReadOnly(bool ro)   { m->readonly=ro; }
void    Editbox::setCursor(int32 cursor){ m->cursor=cursor; }

bool Editbox::handleEvent( const WM_Base &wm )
{
    WidgetHit hit = MouseHitWidget(this, wm);
    bool ret = false;
    if(isVisible()) {
        if(hit==WH_HitMouseLeftDown) {
            setFocus(true);
            ret = true;
        }
        else if(hit==WH_MouseEnter) {
            m->hovered = true;
        }
        else if(hit==WH_MouseLeave) {
            m->hovered = false;
        }
        else if(hit==WH_Nothing) {
            if(wm.type==WMT_IMEResult) {
                if(isFocused()) {
                    setText(WM_IME::cast(wm).text);
                }
            }
        }
    }
    return ret || super::handleEvent(wm);
}



void EditboxMultilineStyle::draw()
{
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}

struct EditboxMultiline::Members
{
    WidgetCallback on_change;
    bool readonly;
    ivec2 cursor;
};
istMemberPtrImpl(EditboxMultiline,Members);

bool EditboxMultiline::isReadOnly() const           { return m->readonly; }
const ivec2& EditboxMultiline::getCursor() const    { return m->cursor; }
Style* EditboxMultiline::createDefaultStyle() const { return istNew(EditboxMultilineStyle)(); }

EditboxMultiline::EditboxMultiline( const wchar_t *text, const WidgetCallback &on_change )
{
    m->on_change = on_change;
    setText(text);
}



} // namespace iui
