#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Title.h"

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
        for(iui::Widget *n=c->getPrevSibling(); n; n=n->getPrevSibling(),++i) {
            if(IsClickable(n)) { return i; }
        }
    }
    i = w->getNumChildren();
    for(iui::Widget *n=w->getLastChild(); n && n!=c; n=n->getPrevSibling(),++i) {
        if(IsClickable(n)) { return i; }
    }
    return i;
}


UICursor::UICursor()
    : m_stack(), m_pos(), m_size(), m_time()
{
}

void UICursor::update(iui::Float dt)
{
    m_time += dt;
    if(atmGetSystemInputs()->isDirectionTriggered(InputState::Dir_Down)) {
        moveNext();
    }
    else if(atmGetSystemInputs()->isDirectionTriggered(InputState::Dir_Up)) {
        movePrev();
    }
    else if(atmGetSystemInputs()->isButtonTriggered(0)) {
        enter();
    }
    else if(atmGetSystemInputs()->isButtonTriggered(1)) {
        cancel();
    }

    if(!m_stack.empty()) {
        State &state = m_stack.back();
        iui::Widget *w = state.widget->getNthChild(state.index);

        m_pos += (w->getPositionAbs()-m_pos) * 0.6f;
        m_size += (w->getSize()-m_size) * 0.6f;
    }
}

void UICursor::draw()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
        if(!state.widget->isVisibleAbs() || IsClickable(state.widget)) {
            return;
        }
    }

    iui::Color bg(1.0f,1.0f,1.5f, 0.5f*glm::sin(glm::radians(m_time)));
    iui::Rect rect(m_pos, m_size);
    iuiGetRenderer()->drawRect(rect, bg);
}

void UICursor::pushSelection(iui::Widget *v, int32 i)
{
    if(v->getTypeID()==iui::WT_List) {
        m_stack.push_back(State(v,i));
    }
    else {
        int32 pos = GetNextClickable(v, i);
        if(pos!=-1) {
            m_stack.push_back(State(v,pos));
        }
    }
}

void UICursor::popSelection()
{
    if(!m_stack.empty()) {
        m_stack.pop_back();
    }
}

void UICursor::clearSelection()
{
    m_stack.clear();
}

void UICursor::setSelection( iui::Widget *v, int32 i )
{
    clearSelection();
    pushSelection(v,i);
}

void UICursor::moveNext()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
        if(state.widget->getTypeID()==iui::WT_List) {
            iui::List *ls = static_cast<iui::List*>(state.widget);
            state.index = ist::wrap<int32>(state.index+1, 0, ls->getItems().size());
        }
        else {
            state.index = GetNextClickable(state.widget, state.index);
            iui::Widget *w = state.widget->getNthChild(state.index);
            if(w->getTypeID()==iui::WT_List) {
                pushSelection(w,0);
            }
        }
    }
}

void UICursor::movePrev()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
        if(state.widget->getTypeID()==iui::WT_List) {
            iui::List *ls = static_cast<iui::List*>(state.widget);
            state.index = ist::wrap<int32>(state.index-1, 0, ls->getItems().size());
        }
        else {
            state.index = GetPrevClickable(state.widget, state.index);
        }
    }
}

void UICursor::enter()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
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

void UICursor::cancel()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
        iui::Widget *w = state.widget;

        iui::WM_Widget wm;
        wm.type = iui::WMT_iuiCancel;
        wm.from = nullptr;
        w->handleEvent(wm);
    }
}


} // namespace atm
