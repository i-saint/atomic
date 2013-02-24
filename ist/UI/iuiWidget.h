#ifndef iui_Widget_h
#define iui_Widget_h
#include "iuiCommon.h"
namespace ist {
namespace iui {


class istInterModule EventHandler
{
public:
    virtual ~EventHandler() {}
};

class istInterModule Widget : public SharedObject
{
public:
    Widget();
    virtual ~Widget();

    virtual void update(Float dt);
    virtual void draw();
    virtual bool handleEvent(const WM_Base &wm);

    WidgetCont&         getChildren();
    const WidgetCont&   getChildren() const;
    Style*              getStyle() const;
    const Position&     getPosition() const;
    const Size&         getSize() const;
    const String&       getText() const;
    bool                isFocused() const;

    void                makeFocus();

protected:
    virtual void setupStyle();

private:
    struct Members;
    deep_copy_ptr<Members> m;
};


class istInterModule Style : public SharedObject
{
public:
    Style(Widget *widget);
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
} // namespace ist
#endif // iui_Widget_h
