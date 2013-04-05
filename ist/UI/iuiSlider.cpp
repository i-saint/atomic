#include "iuiPCH.h"
#include "iuiSlider.h"
#include "iuiSystem.h"
#include "iuiSlider.h"
#include "iuiRenderer.h"
#include "iuiUtilities.h"
namespace iui {


struct HSlider::Members
{
    WidgetCallback on_change;
    Range range;
    Float position;
    bool dragging;

    Members() : position(0.0f), dragging(false)
    {
    }
};
istMemberPtrImpl(HSlider,Members);

Float HSlider::getValue() const { return m->position; }

HSlider::HSlider( const WidgetCallback &on_change )
{
    m->on_change = on_change;
}





void VScrollbarStyle::draw()
{
    VScrollbar *w = static_cast<VScrollbar*>(getWidget());
    Rect rect(Position(), w->getSize());
    Color bg = getBGColor();
    iuiGetRenderer()->drawRect(rect, bg);
    iuiGetRenderer()->drawOutlineRect(rect, getBorderColor());

    const Float range = w->getRange();
    const Float pagesize = w->getPageSize();
    const Float value = w->getValue();

    Float hp = pagesize / (range+pagesize);
    Float h  = std::max<Float>(rect.getSize().y*hp, 12.0f);
    Float pp = value / (range);
    Float p  = (rect.getSize().y-h)*pp;
    Rect bar(Position(2,p+2), Size(w->getSize().x-4, h-4));
    iuiGetRenderer()->drawOutlineRect(bar, getBorderColor());
}
iuiImplDefaultStyle(VScrollbar);


struct VScrollbar::Members
{
    Float           value;
    Float           pagesize;
    Float           range;
    Position        bar_pos;
    Size            bar_size;
    bool            bar_hovered;
    bool            bar_draggind;
    WidgetCallback  on_change_value;

    Members() : value(0.0f), pagesize(0.0f), range(0.0f), bar_pos(), bar_size(), bar_hovered(false), bar_draggind(false)
    {
    }
};
istMemberPtrImpl(VScrollbar,Members);

Float       VScrollbar::getValue() const       { return m->value; }
Float       VScrollbar::getPageSize() const    { return m->pagesize; }
Float       VScrollbar::getRange() const       { return m->range; }
Position    VScrollbar::getBarPosition() const { return m->bar_pos; }
Size        VScrollbar::getBarSize() const     { return m->bar_size; }
bool        VScrollbar::isBarHovered() const   { return m->bar_hovered; }
bool        VScrollbar::isBarDragging() const  { return m->bar_draggind; }

void        VScrollbar::setValue(Float v)
{
    m->value = ist::clamp<Float>(0.0f, v, m->range);
    callIfValid(m->on_change_value);
}

void VScrollbar::setPageSize( Float v )
{
    m->pagesize = v;
}
void        VScrollbar::setRange(Float v)
{
    m->range = std::max<Float>(0.0f, v-m->pagesize);
}

VScrollbar::VScrollbar(Widget *parent, const Rect &rect, WidgetCallback on_change_value)
{
    setParent(parent);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    m->on_change_value = on_change_value;
}

VScrollbar::~VScrollbar()
{
}

void VScrollbar::update( Float dt )
{
    HandleMouseHover(this, m->bar_hovered);
    super::update(dt);
}

bool VScrollbar::handleEvent( const WM_Base &wm )
{
    if(wm.type==WMT_MouseMove) {
        if(m->bar_draggind) {
            const WM_Mouse &m = WM_Mouse::cast(wm);
            scroll(m.mouse_move.y);
        }
    }
    else {
        switch(MouseHit(this, wm)) {
        case WH_HitMouseLeftDown:
            m->bar_draggind = true;
            break;
        case WH_HitMouseLeftUp:
        case WH_MissMouseLeftUp:
            m->bar_draggind = false;
            break;
        }
    }
    return super::handleEvent(wm);
}

void VScrollbar::scroll( Float v )
{
    setValue(getValue()+v);
}


} // namespace iui
