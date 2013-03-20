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




class iuiInterModule HScrollbarStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule HScrollbar : public Widget
{
public:
    iuiImplWidget(HScrollbar);
    HScrollbar(Widget *parent, const Rect &rect=Rect(), WidgetCallback on_change=WidgetCallback());
    Float getValue() const;

private:
    istMemberPtrDecl(Members) m;
};



class iuiInterModule VScrollbarStyle : public Style
{
public:
    virtual void draw();
};

class iuiInterModule VScrollbar : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(VScrollbar);
    VScrollbar(Widget *parent, const Rect &rect=Rect(), WidgetCallback on_change_value=WidgetCallback());
    virtual ~VScrollbar();

    void update(Float dt);

    Float       getValue() const;
    Range       getRange() const;
    Float       getPageSize() const;
    Position    getBarPosition() const;
    Size        getBarSize() const;
    bool        isBarHovered() const;
    bool        isBarDragging() const;

    void        setValue(Float v);
    void        setRange(Range v);
    void        setPageSize(Float v);

    void        scroll(Float v); // == setValue(getValue()+v);

protected:
    virtual bool handleEvent(const WM_Base &wm);

private:
    istMemberPtrDecl(Members) m;
};


} // namespace iui
#endif // iui_Slider_h
