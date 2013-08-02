#ifndef iui_Widget_h
#define iui_Widget_h
#include "iuiCommon.h"
#include "iuiTypes.h"
#include "iuiEvent.h"
namespace iui {

class iuiAPI Widget : public SharedObject
{
friend class UISystem;
public:
    Widget();
    virtual WidgetTypeID getTypeID() const=0;

    virtual void update(Float dt);
    virtual void draw();

    // 破棄したい場合、delete / release() の代わりにこれを呼ぶ。
    // 破棄フラグを立て、WMT_WidgetDelete を発行する。
    void destroy();
    bool isDestroyed() const;

    uint32              getID() const;
    Widget*             getParent() const;
    void                addChild(Widget *c);
    void                eraseChild(Widget *c);
    void                breakLink();

    Widget*             getFirstChild() const;
    Widget*             getLastChild() const;
    Widget*             getNextSibling() const;
    Widget*             getPrevSibling() const;
    uint32              getNumChildren() const;
    Widget*             getNthChild(uint32 n) const;

    Style*              getStyle() const;
    const String&       getText() const;
    const Position&     getPosition() const; // 親からの相対座標
    Position            getPositionAbs() const; // 絶対座標
    const Size&         getSize() const;
    Float               getZOrder() const;
    bool                isVisible() const; // 単身の可視性 (親が不可視でも true を返しうる)
    bool                isVisibleAbs() const; // 親の状態まで考慮した可視性 (不可視の親がいたら false)
    bool                isFocused() const;

    virtual void setParent(Widget *p);
    virtual void setStyle(Style *style);
    virtual void setText(const String &text);
    virtual void setPosition(const Position &pos);
    virtual void setSize(const Size &pos);
    virtual void setZOrder(Float v);
    virtual void setVisibility(bool v);
    virtual void setFocus(bool v);

    void setTextHandler(WidgetCallback cb);
    void setPositionHandler(WidgetCallback cb);
    void setSizeHandler(WidgetCallback cb);
    void setZOrderHandler(WidgetCallback cb);
    void setVisibilityHandler(WidgetCallback cb);
    void setFocusHandler(WidgetCallback cb);

    template<class F>
    void eachChildren(const F &f)
    {
        for(Widget *w=getFirstChild(); w;) {
            Widget *c = w;
            w = w->getNextSibling();
            f(c);
        }
    }
    template<class F>
    void eachChildren(const F &f) const { const_cast<Widget*>(this)->eachChildren(f); }

    template<class F>
    void eachChildrenReverse(const F &f)
    {
        for(Widget *w=getLastChild(); w;) {
            Widget *c = w;
            w = w->getPrevSibling();
            f(c);
        }
    }
    template<class F>
    void eachChildrenReverse(const F &f) const { const_cast<Widget*>(this)->eachChildrenReverse(f); }

    virtual bool handleEvent(const WM_Base &wm);
    virtual bool onOK(const WM_Widget &em);
    virtual bool onCancel(const WM_Widget &em);

protected:
    virtual ~Widget();
    using SharedObject::release;
    void callIfValid(const WidgetCallback &v);

    typedef ist::TPoolFactory<WidgetCont, ist::PoolTraitsST<WidgetCont> > WorkspacePool;
    static WorkspacePool& getWorkspacePool();

private:
    uint32      m_id;
    Widget     *m_parent;
    Widget     *m_first_child;
    Widget     *m_last_child;
    Widget     *m_next_sibling;
    Widget     *m_prev_sibling;

    Style      *m_style;
    String      m_text;
    Position    m_pos;
    Size        m_size;
    Float       m_zorder;
    bool        m_visible;
    bool        m_destroyed;

    WidgetCallback m_on_text;
    WidgetCallback m_on_pos;
    WidgetCallback m_on_size;
    WidgetCallback m_on_zorder;
    WidgetCallback m_on_visibility;
    WidgetCallback m_on_focus;
};


class iuiAPI Style : public SharedObject
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
    Float           getTextHSpacing() const;
    Float           getTextVSpacing() const;

    void setWidget(Widget *v);
    void setFontColor(const Color &v);
    void setBGColor(const Color &v);
    void setBorderColor(const Color &v);
    void setTextHAlign(TextHAlign v);
    void setTextVAlign(TextVAlign v);
    void setTextHSpacing(Float v);
    void setTextVSpacing(Float v);

private:
    Widget     *m_widget;
    Color       m_font_color;
    Color       m_bg_color;
    Color       m_border_color;
    TextHAlign  m_text_halign;
    TextVAlign  m_text_valign;
    Float       m_text_hspacing;
    Float       m_text_vspacing;
};

#define iuiImplWidget(WidgetType)\
    virtual WidgetTypeID getTypeID() const { return WT_##WidgetType; }

#define iuiImplDefaultStyle(WidgetType)\
    static Style* Create##WidgetType##Style() { return iuiNew(WidgetType##Style); }\
    struct Register##WidgetType##Style {\
        Register##WidgetType##Style() { Style::getDefaultStyleCreators()[WT_##WidgetType]=&Create##WidgetType##Style; }\
    } g_register_##WidgetType##Style;


} // namespace iui
#endif // iui_Widget_h
