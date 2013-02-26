#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
namespace iui {

struct Widget::Members
{
    WidgetCont  children;
    Widget      *parent;
    Style       *style;
    Position    pos;
    Size        size;
    String      text;

    Members()
        : parent(NULL)
        , style(NULL)
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
    return false;
}

WidgetCont&         Widget::getChildren()       { return m->children; }
const WidgetCont&   Widget::getChildren() const { return m->children; }
Style*              Widget::getStyle() const    { return m->style; }
const Position&     Widget::getPosition() const { return m->pos; }
const Size&         Widget::getSize() const     { return m->size; }
const String&       Widget::getText() const     { return m->text; }
bool                Widget::isFocused() const   { return iuiGetSystem()->getFocus()==this; }

void Widget::setStyle( Style *style )           { m->style=style; }
void Widget::setText( const String &text )      { m->text = text; }
void Widget::setPosition( const Position &pos ) { m->pos=pos; }
void Widget::setSize( const Size &size )        { m->size=size; }
void Widget::setFocus(bool v)
{
    if(v) {
        iuiGetSystem()->setFocus(this);
    }
    else if(iuiGetSystem()->getFocus()==this) {
        iuiGetSystem()->setFocus(NULL);
    }
}


Style* Widget::createDefaultStyle() const
{
    return NULL;
}



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
