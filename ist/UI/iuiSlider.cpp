#include "iuiPCH.h"
#include "iuiSlider.h"
#include "iuiSystem.h"
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


} // namespace iui
