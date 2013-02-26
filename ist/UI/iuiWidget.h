#ifndef iui_Widget_h
#define iui_Widget_h
#include "iuiCommon.h"
namespace iui {


class iuiInterModule EventHandler
{
public:
    virtual ~EventHandler() {}
};

class iuiInterModule Widget : public SharedObject
{
public:
    Widget();
    virtual ~Widget();

    virtual void update(Float dt);
    virtual void draw();
    virtual bool handleEvent(const WM_Base &wm);

    WidgetCont&         getChildren();
    const WidgetCont&   getChildren() const;
    template<class F>
    void eachChildren(const F &f)
    {
        std::for_each(getChildren().begin(), getChildren().end(), f);
    }

    Style*              getStyle() const;
    const String&       getText() const;
    const Position&     getPosition() const;
    const Size&         getSize() const;
    Float               getZOrder();
    bool                isFocused() const;

    void setStyle(Style *style);
    void setText(const String &text);
    void setPosition(const Position &pos);
    void setSize(const Size &pos);
    void setZOrder(Float v);
    void setFocus(bool v);

protected:
    virtual Style* createDefaultStyle() const;

private:
    struct Members;
    deep_copy_ptr<Members> m;
};


class iuiInterModule Style : public SharedObject
{
public:
    Style();
    virtual ~Style();
    virtual void draw()=0;

    Widget* getWidget() const;
    const Color& getFontColor() const;
    const Color& getBGColor() const;
    const Color& getBorderColor() const;

    void setWidget(Widget *v);
    void setFontColor(const Color &v);
    void setBGColor(const Color &v);
    void setBorderColor(const Color &v);

private:
    struct Members;
    deep_copy_ptr<Members> m;
};

} // namespace iui
#endif // iui_Widget_h
