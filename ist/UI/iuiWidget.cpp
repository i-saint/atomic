#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
#include "iuiUtilities.h"
namespace iui {

uint32              Widget::getID() const           { return m_id; }
Widget*             Widget::getParent() const       { return m_parent; }
Style*              Widget::getStyle() const        { return m_style; }
const Position&     Widget::getPosition() const     { return m_pos; }
const Size&         Widget::getSize() const         { return m_size; }
const String&       Widget::getText() const         { return m_text; }
Float               Widget::getZOrder() const       { return m_zorder; }
bool                Widget::isVisible() const       { return m_visible;  }
bool Widget::isVisibleAbs() const
{
    for(const Widget *p=this; p; p=p->getParent()) {
        if(!p->m_visible) { return false; }
    }
    return true;
}

bool                Widget::isFocused() const   { return iuiGetSystem()->getFocus()==this; }

void Widget::setParent( Widget *p )
{
    if(m_parent) { m_parent->eraseChild(this); }
    m_parent = p;
    if(m_parent) { m_parent->addChild(this); }
}
void Widget::setStyle(Style *style)                   { m_style=style; }
void Widget::setText(const String &text, bool e)      { m_text=text; if(e)callIfValid(m_on_text);        }
void Widget::setPosition(const Position &pos, bool e) { m_pos=pos;   if(e)callIfValid(m_on_pos);         }
void Widget::setSize(const Size &size, bool e)        { m_size=size; if(e)callIfValid(m_on_size);        }
void Widget::setZOrder(float v, bool e)               { m_zorder=v;  if(e)callIfValid(m_on_zorder);      }
void Widget::setVisibility(bool v, bool e)            { m_visible=v; if(e)callIfValid(m_on_visibility);  }
void Widget::setFocus(bool v, bool e)
{
    if(v) {
        iuiGetSystem()->setFocus(this);
    }
    else if(iuiGetSystem()->getFocus()==this) {
        iuiGetSystem()->setFocus(nullptr);
    }
    if(e)callIfValid(m_on_focus);
}

void Widget::setTextHandler(WidgetCallback cb)      { m_on_text=cb; }
void Widget::setPositionHandler(WidgetCallback cb)  { m_on_pos=cb; }
void Widget::setSizeHandler(WidgetCallback cb)      { m_on_size=cb; }
void Widget::setZOrderHandler(WidgetCallback cb)    { m_on_zorder=cb; }
void Widget::setVisibilityHandler(WidgetCallback cb){ m_on_visibility=cb; }
void Widget::setFocusHandler(WidgetCallback cb)     { m_on_focus=cb; }


Widget::Widget()
    : m_id(0)
    , m_parent(nullptr)
    , m_first_child(nullptr)
    , m_last_child(nullptr)
    , m_next_sibling(nullptr)
    , m_prev_sibling(nullptr)
    , m_style(nullptr)
    , m_zorder(0.0f)
    , m_visible(true)
    , m_destroyed(false)
{
    static uint32 s_idgen;
    m_id = ++s_idgen;
    iuiGetSystem()->notifyNewWidget(this);
}

Widget::~Widget()
{
    if(Widget *w=getParent()) { w->eraseChild(this); }
    eachChildren([&](Widget *&w){ w->release(); });
    istSafeRelease(m_style);
}

void Widget::destroy()
{
    if(!m_destroyed) {
        m_destroyed = true;
        WM_Widget wm;
        wm.type = (ist::WMType)WMT_iuiDelete;
        wm.from = this;
        iuiGetSystem()->sendMessage(wm);
    }
}

bool Widget::isDestroyed() const { return m_destroyed; }

void Widget::update(Float dt)
{
    uint32 num_destroyed = 0;
    eachChildren([&](Widget *&w){
        if(w->isDestroyed()) {
            w->release();
            w = nullptr;
            ++num_destroyed;
        }
    });
}

void Widget::draw()
{
    if(m_style==nullptr) {
        m_style = Style::createDefaultStyle(getTypeID());
        if(m_style) { m_style->setWidget(this); }
    }
    if(m_style) {
        m_style->draw();
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
    case WMT_iuiOK:     return onOK(WM_Widget::cast(wm));
    case WMT_iuiCancel: return onCancel(WM_Widget::cast(wm));
    }
    return false;
}
bool Widget::onOK(const WM_Widget &em)     { return false; }
bool Widget::onCancel(const WM_Widget &em) { return false; }

void Widget::callIfValid(const WidgetCallback &v) { if(v){ v(this); } }

void Widget::addChild( Widget *c )
{
    if(c==nullptr) { return; }
    if(!m_first_child) {
        m_first_child = m_last_child =c;
    }
    else {
        m_last_child->m_next_sibling = c;
        c->m_prev_sibling = m_last_child;
        m_last_child = c;
    }
}

void Widget::eraseChild( Widget *c )
{
    if(c==nullptr) { return; }
    if(c==m_first_child) {
        m_first_child = c->m_next_sibling;
    }
    if(c==m_last_child) {
        m_last_child = c->m_prev_sibling;
    }
    c->breakLink();
}

void Widget::breakLink()
{
    if(m_next_sibling) {
        m_next_sibling->m_prev_sibling = m_prev_sibling;
    }
    if(m_prev_sibling) {
        m_prev_sibling->m_next_sibling = m_next_sibling;
    }
}

Widget* Widget::getFirstChild() const   { return m_first_child; }
Widget* Widget::getLastChild() const    { return m_last_child; }
Widget* Widget::getNextSibling() const  { return m_next_sibling; }
Widget* Widget::getPrevSibling() const  { return m_prev_sibling; }
uint32  Widget::getNumChildren() const  {
    uint32 n = 0;
    for(Widget *w=m_first_child; w; w=w->m_next_sibling) {
        ++n;
    }
    return n;
}

Widget* Widget::getNthChild( uint32 n ) const
{
    Widget *r = m_first_child;
    for(uint32 i=0; i<n; ++i) {
        if(r==nullptr) { return nullptr; }
        r = r->m_next_sibling;
    }
    return r;
}

Position Widget::getPositionAbs() const
{
    Position pos;
    for(const Widget *p=this; p; p=p->getParent()) {
        pos += p->getPosition();
    }
    return pos;
}

istImplPoolFunction(Widget::WorkspacePool, Widget::getWorkspacePool, "Widget::WorkspacePool");




Style::Style()
    : m_widget(NULL)
    , m_font_color(1.0f, 1.0f, 1.0f, 1.0f)
    , m_bg_color(0.0f, 0.0f, 0.0f, 0.5f)
    , m_border_color(1.0f, 1.0f, 1.0f, 0.5f)
    , m_text_halign(TA_HLeft)
    , m_text_valign(TA_VCenter)
    , m_text_hspacing(0.75f)
    , m_text_vspacing(1.0f)
{
}

Style::~Style()
{
}

Widget*         Style::getWidget() const        { return m_widget; }
const Color&    Style::getFontColor() const     { return m_font_color; }
const Color&    Style::getBGColor() const       { return m_bg_color; }
const Color&    Style::getBorderColor() const   { return m_border_color; }
TextHAlign      Style::getTextHAlign() const    { return m_text_halign; }
TextVAlign      Style::getTextVAlign() const    { return m_text_valign; }
Float           Style::getTextHSpacing() const  { return m_text_hspacing; }
Float           Style::getTextVSpacing() const  { return m_text_vspacing; }

void Style::setWidget(Widget *v)            { m_widget=v; }
void Style::setFontColor(const Color &v)    { m_font_color=v; }
void Style::setBGColor(const Color &v)      { m_bg_color=v; }
void Style::setBorderColor(const Color &v)  { m_border_color=v; }
void Style::setTextHAlign(TextHAlign v)     { m_text_halign=v; }
void Style::setTextVAlign(TextVAlign v)     { m_text_valign=v; }
void Style::setTextHSpacing(Float v)        { m_text_hspacing=v; }
void Style::setTextVSpacing(Float v)        { m_text_vspacing=v; }

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
