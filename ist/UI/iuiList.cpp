#include "iuiPCH.h"
#include "iuiList.h"
#include "iuiSystem.h"
#include "iuiRenderer.h"
#include "iuiUtilities.h"
#include "iuiSlider.h"
#include "iuiButton.h"
namespace iui {



ListItem::ListItem(const String &text, void *userdata)
    : m_parent(nullptr), m_userdata(nullptr), m_index(0), m_hovered(false), m_selected(false), m_destroyed(false)
{
    m_text = text;
    m_userdata = userdata;
}

ListItem::~ListItem()
{
}

void ListItem::update(Float dt)
{
}

void            ListItem::destroy()                 { m_destroyed=true; }

const String&   ListItem::getText() const           { return m_text; }
void*           ListItem::getUserData() const       { return m_userdata; }
int32           ListItem::getIndex() const          { return m_index; }
bool            ListItem::isHovered() const         { return m_hovered; }
bool            ListItem::isSelected() const        { return m_selected; }
bool            ListItem::isDestroyed() const       { return m_destroyed; }

void            ListItem::setText(const String &v)  { m_text=v; }
void            ListItem::setUserData(void *v)      { m_userdata=v; }
void            ListItem::setIndex(int32 v)         { m_index=v; }
void            ListItem::setHovered(bool v)        { m_hovered=v; }
void            ListItem::setSelected(bool v)       { m_selected=v; }


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



Float               List::getItemHeight() const { return m_item_height; }
Float               List::getScrollPos() const  { return m_scroll_pos; }
VScrollbar*         List::getScrollbar() const  { return m_scrollbar; }
ListItemCont&       List::getItems()            { return m_items; }
const ListItemCont& List::getItems() const      { return m_items; }
Size List::getSizeWithoutScrollbar() const
{
    Size size = getSize();
    size.x -= m_scrollbar->getSize().x;
    return size;
}

void List::setScrollPos(Float v)    {
    Float lim = std::max<Float>(0.0f, getItems().size()*getItemHeight()-getSize().y);
    m_scroll_pos = ist::clamp<Float>(v, 0.0f, lim);
    m_scrollbar->setValue(m_scroll_pos);
}

void List::setItemClickHandler(WidgetCallback cb)       { m_on_item_click=cb; }
void List::setItemDoubleClickHandler(WidgetCallback cb) { m_on_item_doubleclick=cb; }
void List::setItemHoverHandler(WidgetCallback cb)       { m_on_item_hovered=cb; }

List::List( Widget *parent, const Rect &rect, WidgetCallback on_item_click )
    : m_scrollbar(nullptr), m_item_height(18.0f), m_scroll_pos(0.0f)
{
    std::fill_n(m_scroll_buttons, _countof(m_scroll_buttons), (Button*)nullptr);
    setParent(parent);
    setPosition(rect.getPosition());
    setSize(rect.getSize());
    m_on_item_click = on_item_click;

    using std::placeholders::_1;
    Float sb_w = 18.0f;
    Rect scrollbar_rect(Position(rect.getSize().x-sb_w, sb_w), Size(sb_w, rect.getSize().y-(sb_w*2.0f)));
    m_scrollbar = iuiNew(VScrollbar)(this, scrollbar_rect, std::bind(&List::onScroll, this, _1));
    m_scrollbar->setRange(0.0f);
    m_scrollbar->setPageSize(rect.getSize().y);

    m_scroll_buttons[0] = iuiNew(Button)(this, L"△", Rect(Position(rect.getSize().x-sb_w, 0.0f), Size(sb_w,sb_w)), std::bind(&List::onScrollButton, this, _1));
    m_scroll_buttons[1] = iuiNew(Button)(this, L"▽", Rect(Position(rect.getSize().x-sb_w, rect.getSize().y-sb_w), Size(sb_w,sb_w)), std::bind(&List::onScrollButton, this, _1));
    setupDefaultParams();
}

List::~List()
{
    clearItems();
}

void List::update(Float dt)
{
    eachListItem([&](ListItem *item){ item->setHovered(false); });
    bool hovered = false;
    HandleMouseHover(Rect(getPositionAbs(), getSizeWithoutScrollbar()), hovered);
    if(hovered) {
        Position pos = getPositionAbs(); pos.y-=m_scroll_pos;
        Position rel = iuiGetSystem()->getMousePos() - pos;
        int32 index = int32(rel.y / m_item_height);
        if(index>=0 && index<(int32)m_items.size()) {
            m_items[index]->setHovered(true);
            callIfValid(m_on_item_hovered);
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
        m_items.erase(std::remove(m_items.begin(), m_items.end(), (ListItem*)NULL), m_items.end());
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

void List::clearItems()
{
    eachListItem([&](ListItem *item){ item->release(); });
    m_items.clear();
}

void List::addListItem(ListItem *item, int32 pos)
{
    if(pos<0) {
        pos = (int32)m_items.size()+pos+1;
    }
    m_items.insert(m_items.begin()+pos, item);
    onChangeNumItems();
}

void List::addListItem(const String &text, void *userdata, int32 pos)
{
    addListItem(iuiNew(ListItem)(text, userdata), pos);
}

bool List::handleEvent( const WM_Base &wm )
{
    switch(MouseHit(Rect(getPositionAbs(), getSizeWithoutScrollbar()), wm)) {
    case WH_HitMouseLeftDown:
        {
            Position pos = getPositionAbs(); pos.y-=m_scroll_pos;
            Position rel = WM_Mouse::cast(wm).mouse_pos - pos;
            int32 index = int32(rel.y / m_item_height);
            if(index>=0 && index<(int32)m_items.size()) {
                selectItem(index);
            }
            setFocus(true);
            return true;
        }
    case WH_HitMouseWheelUp:
        {
            m_scrollbar->scroll(m_item_height*-2.0f);
            return true;
        }
    case WH_HitMouseWheelDown:
        {
            m_scrollbar->scroll(m_item_height*2.0f);
            return true;
        }
    }
    return super::handleEvent(wm);
}

bool List::selectItem( uint32 index )
{
    eachListItem([&](ListItem *item){ item->setSelected(false); });
    if(index<m_items.size()) {
        m_items[index]->setSelected(true);
        callIfValid(m_on_item_click);
        return true;
    }
    return false;
}

bool List::onOK(const WM_Widget &wm)
{
    return selectItem(wm.option);
}

void List::onChangeNumItems()
{
    Float scroll_size = m_items.size() * m_item_height;
    m_scrollbar->setRange(scroll_size);
}

void List::onScroll( Widget* )
{
    m_scroll_pos = m_scrollbar->getValue();
}

void List::onScrollButton( Widget *w )
{
    float val = 0.0f;
    if(w==m_scroll_buttons[0]) {
        val = m_item_height * -2.0f;
    }
    else if(w==m_scroll_buttons[1]) {
        val = m_item_height * 2.0f;
    }
    m_scrollbar->scroll(val);
}

} // namespace iui
