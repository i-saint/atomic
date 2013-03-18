#ifndef iui_List_h
#define iui_List_h
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



class ListItem;
typedef ist::vector<ListItem*> ListItemCont;

class iuiInterModule ListItem : public SharedObject
{
friend class List;
public:
    ListItem(const String &text=L"", void *userdata=NULL);
    virtual ~ListItem();

    virtual void    update(Float dt);
    void            destroy();

    const String&   getText() const;
    void*           getUserData() const;
    int32           getIndex() const;
    bool            isHovered() const;
    bool            isSelected() const;
    bool            isDestroyed() const;

    void            setText(const String &v);
    void            setUserData(void *v);
private:
    void            setIndex(int32 v);
    void            setHovered(bool v);
    void            setSelected(bool v);

private:
    istMemberPtrDecl(Members) m;
};

class iuiInterModule ListStyle : public Style
{
public:
    ListStyle();
    virtual void draw();
private:
};

class iuiInterModule List : public Widget
{
typedef Widget super;
public:
    iuiImplWidget(List)

    List(Widget *parent, const Rect &rect=Rect(), const WidgetCallback &on_item_click=WidgetCallback());
    ~List();

    void update(Float dt);

    void addListItem(ListItem *item, int32 pos=-1); // -1: 後ろに追加
    void addListItem(const String &text, void *userdata, int32 pos=-1);

    void setItemClickHandler(WidgetCallback cb);
    void setItemDoubleClickHandler(WidgetCallback cb);
    void setItemHoverHandler(WidgetCallback cb);

    ListItemCont&       getItems();
    const ListItemCont& getItems() const;
    Float               getItemHeight() const;
    Float               getScrollPos() const;

    template<class F>
    void eachListItem(const F &f)
    {
        std::for_each(getItems().begin(), getItems().end(), f);
    }

    template<class F>
    void eachListItem(const F &f) const
    {
        std::for_each(getItems().begin(), getItems().end(), f);
    }

protected:
    virtual bool handleEvent(const WM_Base &wm);

private:
    istMemberPtrDecl(Members) m;
};

} // namespace iui
#endif // iui_List_h
