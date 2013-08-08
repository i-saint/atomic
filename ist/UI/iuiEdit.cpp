#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"
#include "iuiEdit.h"
#include "iuiUtilities.h"
namespace iui {



LabelStyle::LabelStyle()
{
}

void LabelStyle::draw()
{
    Label *w = static_cast<Label*>(getWidget());
    Rect rect(Position(), w->getSize());
    Color bg = getBGColor();
    TextPosition tpos(rect, getTextHAlign(), getTextVAlign(), getTextHSpacing(), getTextVSpacing());
    //iuiGetRenderer()->drawRect(rect, bg);
    iuiGetRenderer()->drawFont(tpos, getFontColor(), w->getText().c_str(), w->getText().size());
}
iuiImplDefaultStyle(Label);


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
    Rect rect(Position(), w->getSize());
    TextPosition tpos(rect, getTextHAlign(), getTextVAlign(), getTextHSpacing(), getTextVSpacing());
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
    if(w->isFocused()) {
        vec2 tsize = iuiGetRenderer()->computeTextSize(w->getText().c_str(), w->getCursorPos());
        Line l(Position(tsize.x, 0.0f), Position(tsize.x, tsize.y));
        iuiGetRenderer()->drawLine(l, bg);
    }
}
iuiImplDefaultStyle(Editbox);

Editbox::Editbox(Widget *parent, const wchar_t *text, const Rect &rect, WidgetCallback on_change)
    : m_on_chnage(on_change), m_readonly(false), m_hovered(false), m_ime_chars(0), m_cursor(0)
{
    setParent(parent);
    setText(text);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
}

int32   Editbox::getCursorPos() const      { return m_cursor; }

bool    Editbox::isHovered() const      { return m_hovered; }
bool    Editbox::isReadOnly() const     { return m_readonly; }
void    Editbox::setReadOnly(bool ro)   { m_readonly=ro; }
void    Editbox::setCursor(int32 cursor){ m_cursor=cursor; }

void Editbox::update( Float dt )
{
    HandleMouseHover(this, m_hovered);
    super::update(dt);
}

bool Editbox::handleEvent( const WM_Base &wm )
{
    if(wm.type==WMT_IMEResult) {
        if(isFocused() && !isReadOnly()) {
            const WM_IME &m = WM_IME::cast(wm);
            String text = getText();
            text.insert(text.begin()+m_cursor, m.text, m.text+m.text_len);
            m_cursor+=m.text_len;
            m_ime_chars = m.text_len;
            setText(text);
            return true;
        }
    }
    else if(wm.type==WMT_IMEBegin) {
    }
    else if(wm.type==WMT_IMEEnd) {
    }
    else if(wm.type==WMT_KeyChar) {
        if(!m_ime_chars && isFocused()) {
            const WM_Keyboard &m = WM_Keyboard::cast(wm);
            wchar_t c = (wchar_t)m.key;
            String text = getText();
            bool changed = false;
            if(isprint(c)) {
                text.insert(text.begin()+m_cursor, &c, &c+1);
                ++m_cursor;
                changed = true;
            }
            if(changed) { setText(text); }
            return true;
        }
        if(m_ime_chars>0) {
            --m_ime_chars;
        }
    }
    else if(wm.type==WMT_KeyDown) {
        if(!m_ime_chars && isFocused()) {
            const WM_Keyboard &m = WM_Keyboard::cast(wm);
            wchar_t c = (wchar_t)m.key;
            String text = getText();
            bool changed = false;
            if(c==ist::KEY_ENTER) {
                callIfValid(m_on_chnage);
                setFocus(false);
            }
            else if(c==ist::KEY_DELETE) {
                if(m_cursor<(int32)text.size()) {
                    text.erase(text.begin()+m_cursor, text.begin()+m_cursor+1);
                    changed = true;
                }
            }
            else if(c==ist::KEY_BACK) {
                if(m_cursor>0) {
                    text.erase(text.begin()+m_cursor-1, text.begin()+m_cursor);
                    --m_cursor;
                    changed = true;
                }
            }
            else if(c==ist::KEY_RIGHT) {
                m_cursor = ist::clamp<int32>(m_cursor+1, 0, text.size());
            }
            else if(c==ist::KEY_LEFT) {
                m_cursor = ist::clamp<int32>(m_cursor-1, 0, text.size());
            }
            if(changed) { setText(text); }
        }
    }
    return super::handleEvent(wm);
}

void Editbox::setText( const String &v, bool e )
{
    super::setText(v,e);
    m_cursor = ist::clamp<int32>(m_cursor, 0, v.size());
}

bool Editbox::onOK( const WM_Widget &wm )
{
    setFocus(true);
    return true;
}

bool Editbox::onCancel( const WM_Widget &wm )
{
    setFocus(false);
    return true;
}



void EditboxMultilineStyle::draw()
{
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}


bool EditboxMultiline::isReadOnly() const           { return m_readonly; }
const ivec2& EditboxMultiline::getCursor() const    { return m_cursor; }
Style* EditboxMultiline::createDefaultStyle() const { return iuiNew(EditboxMultilineStyle)(); }

EditboxMultiline::EditboxMultiline( const wchar_t *text, WidgetCallback on_change )
    : m_readonly(false), m_cursor()
{
    setText(text);
    setTextHandler(on_change);
}



} // namespace iui
