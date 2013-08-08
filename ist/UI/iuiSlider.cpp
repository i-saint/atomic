#include "iuiPCH.h"
#include "iuiSlider.h"
#include "iuiSystem.h"
#include "iuiSlider.h"
#include "iuiRenderer.h"
#include "iuiUtilities.h"
namespace iui {


Float HSlider::getValue() const { return m_position; }

HSlider::HSlider( const WidgetCallback &on_change )
    : m_position(0.0f), m_dragging(false)
{
    m_on_change = on_change;
}




void VScrollbarStyle::draw()
{
    VScrollbar *w = static_cast<VScrollbar*>(getWidget());
    Rect rect(Position(), w->getSize());
    Color bg = getBGColor();
    iuiGetRenderer()->drawRect(rect, bg);
    iuiGetRenderer()->drawOutlineRect(rect, getBorderColor());

    Rect bar = w->getBarRect();
    Color bar_color = bg;;
    if     (w->isBarDragging()) { bar_color=Color(1,1,1,0.2); }
    else if(w->isBarHovered())  { bar_color=Color(1,1,1,0.1); }
    iuiGetRenderer()->drawRect(bar, bar_color);
    iuiGetRenderer()->drawOutlineRect(bar, getBorderColor());
}
iuiImplDefaultStyle(VScrollbar);


Float       VScrollbar::getValue() const       { return m_value; }
Float       VScrollbar::getPageSize() const    { return m_pagesize; }
Float       VScrollbar::getRange() const       { return m_range; }
Position    VScrollbar::getBarPosition() const { return m_bar_pos; }
Size        VScrollbar::getBarSize() const     { return m_bar_size; }

Rect VScrollbar::getBarRect() const
{
    Rect rect(getPosition(), getSize());
    const Float range    = getRange();
    const Float pagesize = getPageSize();
    const Float value    = getValue();

    Float hp = pagesize / (range+pagesize);
    Float h  = std::max<Float>(rect.getSize().y*hp, 12.0f);
    Float pp = value / (range);
    Float p  = (rect.getSize().y-h)*pp;
    Rect bar(Position(2,p+2), Size(getSize().x-4, h-4));
    return bar;
}

bool        VScrollbar::isBarHovered() const   { return m_bar_hovered; }
bool        VScrollbar::isBarDragging() const  { return m_bar_draggind; }

void        VScrollbar::setValue(Float v)
{
    m_value = ist::clamp<Float>(0.0f, v, m_range);
    callIfValid(m_on_change_value);
}

void VScrollbar::setPageSize( Float v )
{
    m_pagesize = v;
}
void        VScrollbar::setRange(Float v)
{
    m_range = std::max<Float>(0.0f, v-m_pagesize);
}

VScrollbar::VScrollbar(Widget *parent, const Rect &rect, WidgetCallback on_change_value)
    : m_value(0.0f), m_pagesize(0.0f), m_range(0.0f), m_bar_pos(), m_bar_size(), m_bar_hovered(false), m_bar_draggind(false)
{
    setParent(parent);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    m_on_change_value = on_change_value;
    setupDefaultParams();
}

VScrollbar::~VScrollbar()
{
}

void VScrollbar::update( Float dt )
{
    Rect bar = getBarRect();
    bar.setPosition(bar.getPosition()+getPositionAbs());
    HandleMouseHover(bar, m_bar_hovered);
    super::update(dt);
}

bool VScrollbar::handleEvent( const WM_Base &wm )
{
    if(wm.type==WMT_MouseMove) {
        if(m_bar_draggind) {
            const WM_Mouse &m = WM_Mouse::cast(wm);
            Rect bar = getBarRect();
            Float scroll_range = getSize().y - bar.getSize().y;
            scroll(m.mouse_move.y * (getRange()/scroll_range));
        }
    }
    else {
        Rect bar = getBarRect();
        bar.setPosition(bar.getPosition()+getPositionAbs());
        switch(MouseHit(bar, wm)) {
        case WH_HitMouseLeftDown:
            m_bar_draggind = true;
            break;
        case WH_HitMouseLeftUp:
        case WH_MissMouseLeftUp:
            m_bar_draggind = false;
            break;
        }
        switch(MouseHit(this, wm)) {
        case WH_HitMouseWheelUp:
            {
                scroll(-30.0f);
                return true;
            }
        case WH_HitMouseWheelDown:
            {
                scroll(30.0f);
                return true;
            }
        }
    }
    return super::handleEvent(wm);
}

void VScrollbar::scroll( Float v )
{
    setValue(getValue()+v);
}


} // namespace iui
