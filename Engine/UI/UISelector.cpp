#include "atmPCH.h"
#include "Engine/Game/AtomicApplication.h"
#include "Engine/Game/AtomicGame.h"
#include "UISelector.h"

namespace atm {

bool IsClickable(iui::Widget *w)
{
    switch(w->getTypeID()) {
    case iui::WT_Button:
    case iui::WT_ToggleButton:
    case iui::WT_Checkbox:
    case iui::WT_Editbox:
    case iui::WT_EditboxMultiline:
    case iui::WT_List:
        return true;
    default:
        return false;
    }
}

int32 GetNextClickable(iui::Widget *w, int32 ci)
{
    iui::Widget *c = w->getNthChild(ci);

    int i = ci+1;
    if(c) {
        for(iui::Widget *n=c->getNextSibling(); n; n=n->getNextSibling(),++i) {
            if(IsClickable(n)) { return i; }
        }
    }
    i = 0;
    for(iui::Widget *n=w->getFirstChild(); n && n!=c; n=n->getNextSibling(),++i) {
        if(IsClickable(n)) { return i; }
    }
    return i;
}

int32 GetPrevClickable(iui::Widget *w, int32 ci)
{
    iui::Widget *c = w->getNthChild(ci);

    int i = ci-1;
    if(c) {
        for(iui::Widget *n=c->getPrevSibling(); n; n=n->getPrevSibling(),--i) {
            if(IsClickable(n)) { return i; }
        }
    }
    i = w->getNumChildren()-1;
    for(iui::Widget *n=w->getLastChild(); n && n!=c; n=n->getPrevSibling(),--i) {
        if(IsClickable(n)) { return i; }
    }
    return i;
}


UISelector::UISelector()
    : m_selection(), m_pos(), m_size(), m_time()
{
}

void UISelector::update(iui::Float dt)
{
    m_time += dt;
    if(atmGetMenuInput().isDirectionTriggered(InputState::Dir_Down)) {
        moveNext();
    }
    else if(atmGetMenuInput().isDirectionTriggered(InputState::Dir_Up)) {
        movePrev();
    }
    else if(atmGetMenuInput().isButtonTriggered(0)) {
        enter();
    }
    else if(atmGetMenuInput().isButtonTriggered(1)) {
        cancel();
    }

    updatePosition(0.6f);
}

void UISelector::updatePosition( float32 r )
{
    if(!m_selection.empty()) {
        Selection &state = m_selection.back();
        iui::Position pos;
        iui::Size size;
        if(state.widget->getTypeID()==iui::WT_List) {
            iui::List *ls = static_cast<iui::List*>(state.widget);
            pos = ls->getPositionAbs() + iui::Position(0.0f, -ls->getScrollPos()+ls->getItemHeight()*state.index);
            size = iui::Size(ls->getSizeWithoutScrollbar().x, ls->getItemHeight());
        }
        else {
            iui::Widget *w = state.widget->getNthChild(state.index);
            pos = w->getPositionAbs();
            size = w->getSize();
        }
        m_pos += (pos-m_pos) * r;
        m_size += (size-m_size) * r;
    }
}

void UISelector::draw()
{
    if(m_selection.empty()) { return; }
    if(!m_selection.empty()) {
        Selection &state = m_selection.back();
        if(!state.widget->isVisibleAbs()) {
            return;
        }
    }

    iui::Color bg(0.6f,0.6f,1.0f, 0.5f*(glm::sin(glm::radians(m_time*4.0f))*0.5f+0.5f)+0.4f);
    iui::Rect rect(m_pos, m_size);
    iuiGetRenderer()->drawRect(rect, bg);
}

void UISelector::pushSelection(iui::Widget *v, int32 i)
{
    if(v->getTypeID()==iui::WT_List) {
        m_selection.push_back(Selection(v,i));
    }
    else {
        m_selection.push_back(Selection(v,i));
        moveNext();
        if(m_selection.back().index==-1) {
            m_selection.pop_back();
        }
    }

    if(m_selection.size()==1) {
        updatePosition(1.0f);
    }
}

void UISelector::popSelection()
{
    if(!m_selection.empty()) {
        m_selection.pop_back();
    }
}

uint32 UISelector::popSelection( iui::Widget *v )
{
    uint32 ret = 0;
    while(!m_selection.empty() && m_selection.back().widget!=v) {
        m_selection.pop_back();
        ++ret;
    }
    return ret;
}

void UISelector::clearSelection()
{
    m_selection.clear();
}

void UISelector::setSelection( iui::Widget *v, int32 i )
{
    clearSelection();
    pushSelection(v,i);
}

UISelector::SelectionCont& UISelector::getSelection() { return m_selection; }

void UISelector::enterList(bool backward)
{
    Selection &state = m_selection.back();
    iui::Widget *w = state.widget->getNthChild(state.index);
    if(w->getTypeID()==iui::WT_List) {
        iui::List *ls = static_cast<iui::List*>(w);
        if(!ls->getItems().empty()) {
            pushSelection(ls, backward ? (int32)ls->getItems().size()-1 : 0);
            updateListScroll();
        }
    }
}

void UISelector::updateListScroll()
{
    Selection &back = m_selection.back();
    iui::List *ls = static_cast<iui::List*>(back.widget);
    ls->setScrollPos(ls->getItemHeight()*back.index-ls->getSize().y*0.5f);
}

void UISelector::moveNext()
{
    if(!m_selection.empty()) {
        Selection &state = m_selection.back();
        if(state.widget->getTypeID()==iui::WT_List) {
            iui::List *ls = static_cast<iui::List*>(state.widget);
            state.index++;
            if(state.index>=ls->getItems().size()) {
                popSelection();
                Selection &back = m_selection.back();
                back.index = GetNextClickable(back.widget, back.index);
                enterList(false);
            }
            else {
                updateListScroll();
            }
        }
        else {
            state.index = GetNextClickable(state.widget, state.index);
            enterList(false);
        }
    }
}

void UISelector::movePrev()
{
    if(!m_selection.empty()) {
        Selection &state = m_selection.back();
        if(state.widget->getTypeID()==iui::WT_List) {
            iui::List *ls = static_cast<iui::List*>(state.widget);
            state.index--;
            if(state.index<0) {
                popSelection();
                Selection &back = m_selection.back();
                back.index = GetPrevClickable(back.widget, back.index);
                enterList(true);
            }
            else {
                updateListScroll();
            }
        }
        else {
            state.index = GetPrevClickable(state.widget, state.index);
            enterList(true);
        }
    }
}

void UISelector::enter()
{
    if(!m_selection.empty()) {
        Selection &state = m_selection.back();
        if(state.widget->getTypeID()==iui::WT_List) {
            iui::List *ls = static_cast<iui::List*>(state.widget);
            iui::WM_Widget wm;
            wm.type = iui::WMT_iuiOK;
            wm.option = state.index;
            ls->handleEvent(wm);
        }
        else {
            iui::Widget *w = state.widget->getNthChild(state.index);
            iui::WM_Widget wm;
            wm.type = iui::WMT_iuiOK;
            w->handleEvent(wm);
        }
    }
}

void UISelector::cancel()
{
    if(!m_selection.empty()) {
        {
            Selection &state = m_selection.back();
            if(state.widget->getTypeID()==iui::WT_List) {
                m_selection.pop_back();
            }
        }
        {
            Selection &state = m_selection.back();
            iui::WM_Widget wm;
            wm.type = iui::WMT_iuiCancel;
            wm.from = nullptr;
            state.widget->handleEvent(wm);
        }
    }
}


} // namespace atm
