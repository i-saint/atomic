#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"
#include "iuiEdit.h"
namespace iui {


void EditboxStyle::draw()
{
    Widget *w = getWidget();
    iuiGetRenderer()->drawRect(Rect(w->getPosition(), w->getSize()), getBGColor());
    iuiGetRenderer()->drawOutlineRect(Rect(w->getPosition(), w->getSize()), getBorderColor());
}

struct Editbox::Members
{
    WidgetCallback on_change;
    bool readonly;
    int32 cursor;
};
istMemberPtrImpl(Editbox::Members);

Editbox::Editbox( const wchar_t *text, const WidgetCallback &on_change, Style *style )
{
    m->on_change = on_change;
    setText(text);
    setStyle(style);
}

bool Editbox::isReadOnly() const            { return m->readonly; }
int32 Editbox::getCursor() const            { return m->cursor; }
Style* Editbox::createDefaultStyle() const  { return istNew(EditboxStyle)(); }



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
istMemberPtrImpl(EditboxMultiline::Members);

EditboxMultiline::EditboxMultiline( const wchar_t *text, const WidgetCallback &on_change, Style *style )
{
    m->on_change = on_change;
    setText(text);
    setStyle(style);
}

bool EditboxMultiline::isReadOnly() const           { return m->readonly; }
const ivec2& EditboxMultiline::getCursor() const    { return m->cursor; }
Style* EditboxMultiline::createDefaultStyle() const { return istNew(EditboxMultilineStyle)(); }

} // namespace iui
