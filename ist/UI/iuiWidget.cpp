#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
#include "iuiUtilities.h"
namespace iui {

struct Widget::Members
{
    uint32      id;
    WidgetCont  children;
    Widget      *parent;
    Style       *style;
    String      text;
    Position    pos;
    Size        size;
    Float       zorder;
    bool        visible;
    bool        destroyed;

    WidgetCallback on_text;
    WidgetCallback on_pos;
    WidgetCallback on_size;
    WidgetCallback on_zorder;
    WidgetCallback on_visibility;
    WidgetCallback on_focus;

    Members()
        : id(0)
        , parent(NULL)
        , style(NULL)
        , zorder(0.0f)
        , visible(true)
        , destroyed(false)
    {
    }
};
istMemberPtrImpl(Widget,Members)

uint32              Widget::getID() const           { return m->id; }
Widget*             Widget::getParent() const       { return m->parent; }
WidgetCont&         Widget::getChildren()           { return m->children; }
const WidgetCont&   Widget::getChildren() const     { return m->children; }
Style*              Widget::getStyle() const        { return m->style; }
const Position&     Widget::getPosition() const     { return m->pos; }
const Size&         Widget::getSize() const         { return m->size; }
const String&       Widget::getText() const         { return m->text; }
Float               Widget::getZOrder() const       { return m->zorder; }
bool                Widget::isVisible() const       { return m->visible;  }
bool Widget::isVisibleAbs() const
{
    for(const Widget *p=this; p!=NULL; p=p->getParent()) {
        if(!p->m->visible) { return false; }
    }
    return true;
}

bool                Widget::isFocused() const   { return iuiGetSystem()->getFocus()==this; }

void Widget::setParent( Widget *p )
{
    if(m->parent) { m->parent->eraseChild(this); }
    m->parent = p;
    if(m->parent) { m->parent->addChild(this); }
}
void Widget::setStyle( Style *style )           { m->style=style; }
void Widget::setText( const String &text )      { m->text=text; callIfValid(m->on_text);        }
void Widget::setPosition( const Position &pos ) { m->pos=pos;   callIfValid(m->on_pos);         }
void Widget::setSize( const Size &size )        { m->size=size; callIfValid(m->on_size);        }
void Widget::setZOrder(float v)                 { m->zorder=v;  callIfValid(m->on_zorder);      }
void Widget::setVisibility(bool v)              { m->visible=v; callIfValid(m->on_visibility);  }
void Widget::setFocus(bool v)
{
    if(v) {
        iuiGetSystem()->setFocus(this);
    }
    else if(iuiGetSystem()->getFocus()==this) {
        iuiGetSystem()->setFocus(NULL);
    }
    callIfValid(m->on_focus);
}

void Widget::setTextHandler(WidgetCallback cb)      { m->on_text=cb; }
void Widget::setPositionHandler(WidgetCallback cb)  { m->on_pos=cb; }
void Widget::setSizeHandler(WidgetCallback cb)      { m->on_size=cb; }
void Widget::setZOrderHandler(WidgetCallback cb)    { m->on_zorder=cb; }
void Widget::setVisibilityHandler(WidgetCallback cb){ m->on_visibility=cb; }
void Widget::setFocusHandler(WidgetCallback cb)     { m->on_focus=cb; }


Widget::Widget()
{
    static uint32 s_idgen;
    m->id = ++s_idgen;
    iuiGetSystem()->notifyNewWidget(this);
}

Widget::~Widget()
{
    if(Widget *w=getParent()) { w->eraseChild(this); }
    eachChildren([&](Widget *&w){ w->release(); });
    istSafeRelease(m->style);
}

void Widget::destroy()
{
    if(!m->destroyed) {
        m->destroyed = true;
        WM_Widget wm;
        wm.type = (ist::WMType)WMT_iuiDelete;
        wm.from = this;
        iuiGetSystem()->sendMessage(wm);
    }
}

bool Widget::isDestroyed() const { return m->destroyed; }

void Widget::update(Float dt)
{
    uint32 num_destroyed = 0;
    eachChildren([&](Widget *&w){
        if(w->isDestroyed()) {
            w->release();
            w = NULL;
            ++num_destroyed;
        }
    });
    if(num_destroyed>0) {
        m->children.erase(std::remove(m->children.begin(), m->children.end(), (Widget*)NULL), m->children.end());
    }
}

void Widget::draw()
{
    if(m->style==NULL) {
        m->style = Style::createDefaultStyle(getTypeID());
        if(m->style) { m->style->setWidget(this); }
    }
    if(m->style) {
        m->style->draw();
    }
}

bool Widget::handleEvent(const WM_Base &wm)
{
    WidgetHit wh = MouseHit(this, wm);
    switch(wh) {
    case WH_HitMouseLeftDown:
    case WH_HitMouseRightDown:
    case WH_HitMouseMiddleDown:
    case WH_HitMouseLeftUp:
    case WH_HitMouseRightUp:
    case WH_HitMouseMiddleUp:
    case WH_MouseInside:
        if(wh==WH_HitMouseLeftDown && !isFocused()) {
            setFocus(true);
        }
        return true;
    }
    switch(wm.type) {
    case WMT_iuiOK:     return onOK();
    case WMT_iuiCancel: return onCancel();
    }
    return false;
}
bool Widget::onOK()     { return false; }
bool Widget::onCancel() { return false; }

void Widget::callIfValid(const WidgetCallback &v) { if(v){ v(this); } }

void Widget::addChild( Widget *c )
{
    if(c==NULL) { return; }
    m->children.insert(c);
}

void Widget::eraseChild( Widget *c )
{
    if(c==NULL) { return; }
    auto p = std::find(m->children.begin(), m->children.end(), c);
    if(p!=m->children.end()) {
        m->children.erase(p);
    }
}

Position Widget::getPositionAbs() const
{
    Position pos;
    for(const Widget *p=this; p!=nullptr; p=p->getParent()) {
        pos += p->getPosition();
    }
    return pos;
}

istImplPoolFunction(Widget::WorkspacePool, Widget::getWorkspacePool, "Widget::WorkspacePool");




struct Style::Members
{
    Widget *widget;
    Color font_color;
    Color bg_color;
    Color border_color;
    TextHAlign text_halign;
    TextVAlign text_valign;
    Float text_hspacing;
    Float text_vspacing;

    Members()
        : widget(NULL)
        , font_color(1.0f, 1.0f, 1.0f, 1.0f)
        , bg_color(0.0f, 0.0f, 0.0f, 0.5f)
        , border_color(1.0f, 1.0f, 1.0f, 0.5f)
        , text_halign(TA_HLeft)
        , text_valign(TA_VCenter)
        , text_hspacing(0.75f)
        , text_vspacing(1.0f)
    {
    }
};
istMemberPtrImpl(Style,Members)

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
TextHAlign      Style::getTextHAlign() const    { return m->text_halign; }
TextVAlign      Style::getTextVAlign() const    { return m->text_valign; }
Float           Style::getTextHSpacing() const  { return m->text_hspacing; }
Float           Style::getTextVSpacing() const  { return m->text_vspacing; }

void Style::setWidget(Widget *v)            { m->widget=v; }
void Style::setFontColor(const Color &v)    { m->font_color=v; }
void Style::setBGColor(const Color &v)      { m->bg_color=v; }
void Style::setBorderColor(const Color &v)  { m->border_color=v; }
void Style::setTextHAlign(TextHAlign v)     { m->text_halign=v; }
void Style::setTextVAlign(TextVAlign v)     { m->text_valign=v; }
void Style::setTextHSpacing(Float v)        { m->text_hspacing=v; }
void Style::setTextVSpacing(Float v)        { m->text_vspacing=v; }

Style::StyleCreatorTable& Style::getDefaultStyleCreators()
{
    static StyleCreatorTable s_table;
    return s_table;
}

Style* Style::createDefaultStyle( uint32 widget_typeid )
{
    StyleCreatorTable &table = getDefaultStyleCreators();
    //istAssert(table[widget_typeid]);
    if(table[widget_typeid]) {
        return table[widget_typeid]();
    }
    else {
        return NULL;
    }
}


} // namespace iui
