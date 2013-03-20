#include "iuiPCH.h"
#include "iuiSlider.h"
#include "iuiSystem.h"
#include "iuiSlider.h"
#include "iuiRenderer.h"
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
}
iuiImplDefaultStyle(VScrollbar);


struct VScrollbar::Members
{
    Float           value;
    Range           range;
    Float           page_size;
    Position        bar_pos;
    Size            bar_size;
    bool            bar_hovered;
    bool            bar_draggind;
    WidgetCallback  on_change_value;

    Members() : value(0.0f), range(), page_size(0.0f), bar_pos(), bar_size(), bar_hovered(false), bar_draggind(false)
    {
    }
};
istMemberPtrImpl(VScrollbar,Members);

Float       VScrollbar::getValue() const       { return m->value; }
Range       VScrollbar::getRange() const       { return m->range; }
Float       VScrollbar::getPageSize() const    { return m->page_size; }
Position    VScrollbar::getBarPosition() const { return m->bar_pos; }
Size        VScrollbar::getBarSize() const     { return m->bar_size; }
bool        VScrollbar::isBarHovered() const   { return m->bar_hovered; }
bool        VScrollbar::isBarDragging() const  { return m->bar_draggind; }

void        VScrollbar::setValue(Float v)      { m->value=ist::clamp<Float>(m->range.min, v, m->range.max); callIfValid(m->on_change_value); }
void        VScrollbar::setRange(Range v)      { m->range=v; }
void        VScrollbar::setPageSize(Float v)   { m->page_size=v; }

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
    super::update(dt);
}

bool VScrollbar::handleEvent( const WM_Base &wm )
{
    return super::handleEvent(wm);
}

void VScrollbar::scroll( Float v )
{
    setValue(getValue()+v);
}


} // namespace iui
