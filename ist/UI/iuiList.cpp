#include "iuiPCH.h"
#include "iuiList.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"
#include "iuiUtilities.h"
#include "iuiSlider.h"
namespace iui {



struct ListItem::Members
{
    String text;
    void *userdata;
    int32 index;
    bool hovered;
    bool selected;
    bool destroyed;

    Members()
        : userdata(NULL), index(0), hovered(false), selected(false), destroyed(false)
    {}
};
istMemberPtrImpl(ListItem,Members);

ListItem::ListItem(const String &text, void *userdata)
{
    m->text = text;
    m->userdata = userdata;
}

ListItem::~ListItem()
{
}

void ListItem::update(Float dt)
{
}

void            ListItem::destroy()                 { m->destroyed=true; }

const String&   ListItem::getText() const           { return m->text; }
void*           ListItem::getUserData() const       { return m->userdata; }
int32           ListItem::getIndex() const          { return m->index; }
bool            ListItem::isHovered() const         { return m->hovered; }
bool            ListItem::isSelected() const        { return m->selected; }
bool            ListItem::isDestroyed() const       { return m->destroyed; }

void            ListItem::setText(const String &v)  { m->text=v; }
void            ListItem::setUserData(void *v)      { m->userdata=v; }
void            ListItem::setIndex(int32 v)         { m->index=v; }
void            ListItem::setHovered(bool v)        { m->hovered=v; }
void            ListItem::setSelected(bool v)       { m->selected=v; }


ListStyle::ListStyle()
{
    setTextHAlign(TA_HLeft);
    setTextHSpacing(0.75f);
}

void ListStyle::draw()
{
    List *w = static_cast<List*>(getWidget());
    VScrollbar *scrollbar = w->getScrollbar();
    {
        SetupScreen(Rect(w->getPositionAbs(), w->getSizeWithoutScrollbar()));
    }

    Rect rect(Position(), w->getSizeWithoutScrollbar());
    TextPosition tpos(rect, getTextHAlign(), getTextVAlign(), getTextHSpacing(), getTextVSpacing());
    Color bg = getBGColor();
    iuiGetRenderer()->drawRect(rect, bg);
    iuiGetRenderer()->drawOutlineRect(rect, getBorderColor());

    Float item_height = w->getItemHeight();
    Float scrollpos = w->getScrollPos();
    int32 nth_item = 0;
    w->eachListItem([&](ListItem *item){
        Rect irect(Position(0.0f, item_height*nth_item-scrollpos), Size(w->getSize().x, item_height));
        if(IsOverlaped(rect, irect)) { // 表示領域外ならスキップ
            drawItem(item, irect);
        }
        ++nth_item;
    });
}

void ListStyle::drawItem(ListItem *item, const Rect &rect)
{
    TextPosition tpos(rect, getTextHAlign(), getTextVAlign(), getTextHSpacing(), getTextVSpacing());
    if(item->isSelected()) {
        Color ibg = vec4(1.0f, 1.0f, 1.0f, 0.4f);
        iuiGetRenderer()->drawRect(rect, ibg);
    }
    else if(item->isHovered()) {
        Color ibg = vec4(1.0f, 1.0f, 1.0f, 0.2f);
        iuiGetRenderer()->drawRect(rect, ibg);
    }
    const String &text = item->getText();
    iuiGetRenderer()->drawFont(tpos, getFontColor(), text.c_str(), text.size());
}

iuiImplDefaultStyle(List);



struct List::Members
{
    VScrollbar *scrollbar;
    ListItemCont items;
    WidgetCallback on_item_click;
    WidgetCallback on_item_doubleclick;
    WidgetCallback on_item_hovered;
    Float item_height;
    Float scroll_pos;

    Members() : scrollbar(NULL), item_height(18.0f), scroll_pos(0.0f) {}
};
istMemberPtrImpl(List,Members);

Float               List::getItemHeight() const { return m->item_height; }
Float               List::getScrollPos() const  { return m->scroll_pos; }
VScrollbar*         List::getScrollbar() const  { return m->scrollbar; }
ListItemCont&       List::getItems()            { return m->items; }
const ListItemCont& List::getItems() const      { return m->items; }
Size List::getSizeWithoutScrollbar() const
{
    Size size = getSize();
    size.x -= m->scrollbar->getSize().x;
    return size;
}

void List::setItemClickHandler(WidgetCallback cb)       { m->on_item_click=cb; }
void List::setItemDoubleClickHandler(WidgetCallback cb) { m->on_item_doubleclick=cb; }
void List::setItemHoverHandler(WidgetCallback cb)       { m->on_item_hovered=cb; }

List::List( Widget *parent, const Rect &rect, WidgetCallback on_item_click )
{
    setParent(parent);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    m->on_item_click = on_item_click;

    Float scrollbar_width = 14.0f;
    Rect scrollbar_rect(Position(rect.getSize().x-scrollbar_width, 0.0f), Size(scrollbar_width, rect.getSize().y));
    m->scrollbar = istNew(VScrollbar)(this, scrollbar_rect, std::bind(&List::onScroll, this, std::placeholders::_1));
    m->scrollbar->setRange(Range(0.0f, 0.0f));
}

List::~List()
{
    eachListItem([&](ListItem *item){ item->release(); });
}

void List::update(Float dt)
{
    eachListItem([&](ListItem *item){ item->setHovered(false); });
    bool hovered = false;
    HandleMouseHover(Rect(getPositionAbs(), getSizeWithoutScrollbar()), hovered);
    if(hovered) {
        Position pos = getPositionAbs(); pos.y-=m->scroll_pos;
        Position rel = iuiGetSystem()->getMousePos() - pos;
        int32 index = int32(rel.y / m->item_height);
        if(index>=0 && index<(int32)m->items.size()) {
            m->items[index]->setHovered(true);
            callIfValid(m->on_item_hovered);
        }
    }

    uint32 num_destroyed = 0;
    eachListItem([&](ListItem *&item){
        item->update(dt);
        if(item->isDestroyed()) {
            item->release();
            item = NULL;
            ++num_destroyed;
        }
    });
    if(num_destroyed>0) {
        m->items.erase(std::remove(m->items.begin(), m->items.end(), (ListItem*)NULL), m->items.end());
        onChangeNumItems();
    }

    super::update(dt);
}

const ListItem* List::getSelectedItem( size_t i ) const
{
    const ListItem *ret = NULL;
    size_t nth = 0;
    eachListItem([&](const ListItem *item){
        if(item->isSelected() && nth==i) { ret=item; }
    });
    return ret;
}

void List::addListItem(ListItem *item, int32 pos)
{
    if(pos<0) {
        pos = (int32)m->items.size()+pos+1;
    }
    m->items.insert(m->items.begin()+pos, item);
    onChangeNumItems();
}

void List::addListItem(const String &text, void *userdata, int32 pos)
{
    addListItem(istNew(ListItem)(text, userdata), pos);
}

bool List::handleEvent( const WM_Base &wm )
{
    switch(MouseHit(Rect(getPositionAbs(), getSizeWithoutScrollbar()), wm)) {
    case WH_HitMouseLeftDown:
        {
            eachListItem([&](ListItem *item){ item->setSelected(false); });

            Position pos = getPositionAbs(); pos.y-=m->scroll_pos;
            Position rel = WM_Mouse::cast(wm).mouse_pos - pos;
            int32 index = int32(rel.y / m->item_height);
            if(index>=0 && index<(int32)m->items.size()) {
                m->items[index]->setSelected(true);
                callIfValid(m->on_item_click);
            }
            setFocus(true);
            return true;
        }
    case WH_HitMouseWheelUp:
        {
            m->scrollbar->scroll(m->item_height*-2.0f);
            return true;
        }
    case WH_HitMouseWheelDown:
        {
            m->scrollbar->scroll(m->item_height*2.0f);
            return true;
        }
    }
    return super::handleEvent(wm);
}

void List::onChangeNumItems()
{
    Float scroll_size = std::max<Float>(0.0f, m->items.size() * m->item_height - getSize().y);
    m->scrollbar->setRange(Range(0.0f, scroll_size));
}

void List::onScroll( Widget* )
{
    m->scroll_pos = m->scrollbar->getValue();
}

} // namespace iui
