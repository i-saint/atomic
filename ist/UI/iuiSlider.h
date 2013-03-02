#ifndef iui_Slider_h
#define iui_Slider_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {


class iuiInterModule HSliderStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule HSlider : public Widget
{
public:
    iuiImplWidget(HSlider)
    HSlider(const WidgetCallback &on_change);
    Float getValue() const;

private:
    istMemberPtrDecl(Members) m;
};

} // namespace iui
#endif // iui_Slider_h
