#ifndef iui_Slider_h
#define iui_Slider_h
#include "iuiCommon.h"
#include "iuiWidget.h"
namespace iui {


class iuiAPI HSliderStyle : public Style
{
public:
    virtual void draw();
};

class iuiAPI HSlider : public Widget
{
public:
    iuiImplWidget(HSlider)
    HSlider(const WidgetCallback &on_change);
    Float getValue() const;

private:
    WidgetCallback m_on_change;
    Range   m_range;
    Float   m_position;
    bool    m_dragging;
};




class iuiAPI HScrollbarStyle : public Style
{
public:
    virtual void draw();
};

class iuiAPI HScrollbar : public Widget
{
public:
    iuiImplWidget(HScrollbar);
    HScrollbar(Widget *parent, const Rect &rect=Rect(), WidgetCallback on_change=WidgetCallback());
    Float getValue() const;

private:
};



class iuiAPI VScrollbarStyle : public Style
{
public:
    virtual void draw();
};

class iuiAPI VScrollbar : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(VScrollbar);
    VScrollbar(Widget *parent, const Rect &rect=Rect(), WidgetCallback on_change_value=WidgetCallback());
    virtual ~VScrollbar();

    void update(Float dt);

    Float       getValue() const;
    Float       getPageSize() const;
    Float       getRange() const;
    Position    getBarPosition() const;
    Size        getBarSize() const;
    Rect        getBarRect() const;
    bool        isBarHovered() const;
    bool        isBarDragging() const;

    void        setValue(Float v);
    void        setPageSize(Float v);
    void        setRange(Float v);

    void        scroll(Float v); // == setValue(getValue()+v);

    bool handleEvent(const WM_Base &wm) override;

private:
    Float           m_value;
    Float           m_pagesize;
    Float           m_range;
    Position        m_bar_pos;
    Size            m_bar_size;
    bool            m_bar_hovered;
    bool            m_bar_draggind;
    WidgetCallback  m_on_change_value;
};


} // namespace iui
#endif // iui_Slider_h
