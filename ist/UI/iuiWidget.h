#ifndef iui_Widget_h
#define iui_Widget_h
#include "iuiCommon.h"
#include "iuiTypes.h"
#include "iuiEvent.h"
namespace iui {

class iuiInterModule Widget : public SharedObject
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
    WidgetCont&         getChildren();
    const WidgetCont&   getChildren() const;
    void                addChild(Widget *c);
    void                eraseChild(Widget *c);

    Style*              getStyle() const;
    const String&       getText() const;
    const Position&     getPosition() const; // 親からの相対座標
    Position            getPositionAbs() const; // 絶対座標
    const Size&         getSize() const;
    Float               getZOrder() const;
    bool                isVisible() const; // 単身の可視性 (親が不可視でも true を返しうる)
    bool                isVisibleAbs() const; // 親の状態まで考慮した可視性 (不可視の親がいたら false)
    bool                isFocused() const;

    void setParent(Widget *p);
    void setStyle(Style *style);
    void setText(const String &text);
    void setPosition(const Position &pos);
    void setSize(const Size &pos);
    void setZOrder(Float v);
    void setVisibility(bool v);
    void setFocus(bool v);

    void setTextHandler(WidgetCallback cb);
    void setPositionHandler(WidgetCallback cb);
    void setSizeHandler(WidgetCallback cb);
    void setZOrderHandler(WidgetCallback cb);
    void setVisibilityHandler(WidgetCallback cb);
    void setFocusHandler(WidgetCallback cb);

    template<class F>
    void eachChildren(const F &f)
    {
        WidgetCont &children = getChildren();
        if(children.empty()) { return; }
        // 処理中に children の要素数が変動する可能性があるため、ワークスペースにコピーしてそれを処理。
        // コストを伴いますが致し方ありません。
        WidgetCont *workspace = getWorkspacePool().create();
        *workspace = children;
        std::for_each(workspace->begin(), workspace->end(), f);
        getWorkspacePool().recycle(workspace);
    }
    template<class F>
    void eachChildren(const F &f) const { const_cast<Widget*>(this)->eachChildren(f); }

    template<class F>
    void eachChildrenReverse(const F &f)
    {
        WidgetCont &children = getChildren();
        if(children.empty()) { return; }
        WidgetCont *workspace = getWorkspacePool().create();
        *workspace = children;
        std::for_each(workspace->rbegin(), workspace->rend(), f);
        getWorkspacePool().recycle(workspace);
    }
    template<class F>
    void eachChildrenReverse(const F &f) const { const_cast<Widget*>(this)->eachChildrenReverse(f); }

protected:
    virtual ~Widget();
    using SharedObject::release;
    virtual bool handleEvent(const WM_Base &wm);
    void callIfValid(const WidgetCallback &v);

    typedef ist::TPoolFactory<WidgetCont, ist::PoolFactoryTraitsST<WidgetCont> > WorkspacePool;
    static WorkspacePool& getWorkspacePool();

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
