#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
namespace iui {

struct Widget::Members
{
    WidgetCont  children;
    Widget      *parent;
    Style       *style;
    String      text;
    Position    pos;
    Size        size;
    Float       zorder;
    bool        visible;

    WidgetCallback on_text;
    WidgetCallback on_pos;
    WidgetCallback on_size;
    WidgetCallback on_zorder;
    WidgetCallback on_visible;
    WidgetCallback on_focus;

    Members()
        : parent(NULL)
        , style(NULL)
        , zorder(0.0f)
        , visible(true)
    {
    }
};

Widget::Widget()
{
}

Widget::~Widget()
{
}

void Widget::update(Float dt)
{
}

void Widget::draw()
{
    if(m->style==NULL) {
        m->style = createDefaultStyle();
        if(m->style) { m->style->setWidget(this); }
    }
    if(m->style) {
        m->style->draw();
    }
}

bool Widget::handleEvent(const WM_Base &wm)
{
    switch(wm.type) {
    case WMT_Unknown:
        break;
    }
    return false;
}

WidgetCont&         Widget::getChildren()       { return m->children; }
const WidgetCont&   Widget::getChildren() const { return m->children; }
Style*              Widget::getStyle() const    { return m->style; }
const Position&     Widget::getPosition() const { return m->pos; }
const Size&         Widget::getSize() const     { return m->size; }
const String&       Widget::getText() const     { return m->text; }
Float               Widget::getZOrder() const   { return m->zorder; }
bool                Widget::isVisible() const   { return m->visible; }
bool                Widget::isFocused() const   { return iuiGetSystem()->getFocus()==this; }

void Widget::setStyle( Style *style )           { m->style=style; }
void Widget::setText( const String &text )      { m->text=text; CallIfValid(m->on_text); }
void Widget::setPosition( const Position &pos ) { m->pos=pos; CallIfValid(m->on_pos); }
void Widget::setSize( const Size &size )        { m->size=size; CallIfValid(m->on_size); }
void Widget::setZOrder(float v)                 { m->zorder=v; CallIfValid(m->on_zorder); }
void Widget::setVisible(bool v)                 { m->visible=v; CallIfValid(m->on_visible); }
void Widget::setFocus(bool v)
{
    if(v) {
        iuiGetSystem()->setFocus(this);
    }
    else if(iuiGetSystem()->getFocus()==this) {
        iuiGetSystem()->setFocus(NULL);
    }
    CallIfValid(m->on_focus);
}

void Widget::setTextHandler(WidgetCallback cb)      { m->on_text=cb; }
void Widget::setPositionHandler(WidgetCallback cb)  { m->on_pos=cb; }
void Widget::setSizeHandler(WidgetCallback cb)      { m->on_size=cb; }
void Widget::setZOrderHandler(WidgetCallback cb)    { m->on_zorder=cb; }
void Widget::setVisibilityHandler(WidgetCallback cb){ m->on_visible=cb; }
void Widget::setFocusHandler(WidgetCallback cb)     { m->on_focus=cb; }

Style* Widget::createDefaultStyle() const
{
    return NULL;
}

void Widget::CallIfValid(const WidgetCallback &v) { if(v){ v(this); } }



struct Style::Members
{
    Widget *widget;
    Color font_color;
    Color bg_color;
    Color border_color;
};

Style::Style()
{
}

Style::~Style()
{
}

Widget*         Style::getWidget() const        { return m->widget; }
const Color&    Style::getFontColor() const     { return m->font_color; }
const Color&    Style::getBGColor() const       { return m->bg_color; }
const Color&    Style::getBorderColor() const   { return m->border_color; }

void Style::setWidget(Widget *v)            { m->widget=v; }
void Style::setFontColor(const Color &v)    { m->font_color=v; }
void Style::setBGColor(const Color &v)      { m->bg_color=v; }
void Style::setBorderColor(const Color &v)  { m->border_color=v; }


} // namespace iui
