#ifndef iui_Widget_h
#define iui_Widget_h
#include "iuiCommon.h"
namespace ist {
namespace iui {

class Widget;
typedef ist::vector<Widget*> Widgets;
class Style;


class istInterModule EventHandler
{
public:
    virtual ~EventHandler() {}
};

class istInterModule Widget : public SharedObject
{
public:
    Widget();
    ~Widget();

    virtual void update(Float dt);
    virtual bool handleEvent();

    Widgets&        getChildren()       { return m_children; }
    const Widgets&  getChildren() const { return m_children; }
    Style*          getStyle() const    { return m_style; }
    const Position& getPosition() const { return m_pos; }
    const Size&     getSize() const     { return m_size; }
    const String&   getText() const     { return m_text; }

private:
    Widgets m_children;
    Widget *m_parent;
    Style *m_style;
    Position m_pos;
    Size m_size;
    String m_text;
};


class istInterModule Style : public SharedObject
{
public:
    Style();
    virtual ~Style();

    const Widget* getWidget() const { return m_widget; }
    void setWidget(const Widget *w) { m_widget=w; }

    virtual void draw() const;

private:
    const Widget *m_widget;
};

} // namespace iui
} // namespace ist
#endif // iui_Widget_h
