#ifndef iui_List_h
#define iui_List_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {

class iuiInterModule ListItem : public Widget
{
public:
private:
    struct Members
    {
        String text;
        uint32 index;
        bool selected;
        bool hovered;

        ListCallback on_click;
        ListCallback on_doubleclick;
    };
};


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
#endif // iui_List_h
