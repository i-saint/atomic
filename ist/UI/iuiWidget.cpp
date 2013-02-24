#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
namespace ist {
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

void Widget::makeFocus()
{
    iuiGetSystem()->setFocus(this);
}



struct Style::Members
{
    Widget *widget;
    Color font_color;
    Color bg_color;
    Color border_color;
};

Style::Style(Widget *widget)
{
    m->widget = widget;
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
} // namespace ist
