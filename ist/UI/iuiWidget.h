#ifndef iui_Widget_h
#define iui_Widget_h
#include "iuiCommon.h"
#include "iuiTypes.h"
#include "iuiEvent.h"
namespace iui {


class iuiInterModule Widget : public SharedObject
{
public:
    Widget();
    virtual ~Widget();
    virtual WidgetTypeID getTypeID() const=0;

    virtual void update(Float dt);
    virtual void draw();
    virtual bool handleEvent(const WM_Base &wm);

    WidgetCont&         getChildren();
    const WidgetCont&   getChildren() const;
    template<class F>
    void eachChildren(const F &f) { std::for_each(getChildren().begin(), getChildren().end(), f); }
    template<class F>
    void eachChildrenReverse(const F &f) { std::for_each(getChildren().rbegin(), getChildren().rend(), f); }

    Style*              getStyle() const;
    const String&       getText() const;
    const Position&     getPosition() const;
    const Size&         getSize() const;
    Float               getZOrder() const;
    bool                isVisible() const;
    bool                isFocused() const;

    void setStyle(Style *style);
    void setText(const String &text);
    void setPosition(const Position &pos);
    void setSize(const Size &pos);
    void setZOrder(Float v);
    void setVisible(bool v);
    void setFocus(bool v);

    void setTextHandler(WidgetCallback cb);
    void setPositionHandler(WidgetCallback cb);
    void setSizeHandler(WidgetCallback cb);
    void setZOrderHandler(WidgetCallback cb);
    void setVisibilityHandler(WidgetCallback cb);
    void setFocusHandler(WidgetCallback cb);

protected:
    void CallIfValid(const WidgetCallback &v);

private:
    istMemberPtrDecl(Members) m;
};


class iuiInterModule Style : public SharedObject
{
public:
    typedef Style* (*StyleCreator)();
    typedef StyleCreator (StyleCreatorTable)[WT_End];
    static StyleCreatorTable& getDefaultStyleCreators();
    static Style* createDefaultStyle(uint32 widget_typeid);

    Style();
    virtual ~Style();
    virtual void draw()=0;

    Widget*         getWidget() const;
    const Color&    getFontColor() const;
    const Color&    getBGColor() const;
    const Color&    getBorderColor() const;
    TextHAlign      getTextHAlign() const;
    TextVAlign      getTextVAlign() const;

    void setWidget(Widget *v);
    void setFontColor(const Color &v);
    void setBGColor(const Color &v);
    void setBorderColor(const Color &v);
    void setTextHAlign(TextHAlign v);
    void setTextVAlign(TextVAlign v);

private:
    istMemberPtrDecl(Members) m;
};

#define iuiImplWidget(WidgetType)\
    virtual WidgetTypeID getTypeID() const { return WT_##WidgetType; }

#define iuiImplDefaultStyle(WidgetType)\
    static Style* Create##WidgetType##Style() { return istNew(WidgetType##Style); }\
    struct Register##WidgetType##Style {\
        Register##WidgetType##Style() { Style::getDefaultStyleCreators()[WT_##WidgetType]=&Create##WidgetType##Style; }\
    } g_register_##WidgetType##Style;


} // namespace iui
#endif // iui_Widget_h
