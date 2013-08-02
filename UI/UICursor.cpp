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
        return true;
    default:
        return false;
    }
}

int32 GetNextClickable(iui::Widget *w, int32 ci)
{
    iui::WidgetCont &cont = w->getChildren();
    if(ci>=0 && ci>=cont.size()) { return -1; }

    for(int32 i=ci+1; i<cont.size(); ++i) {
        if(IsClickable(cont[i])) { return i; }
    }
    for(int32 i=0; i<ci; ++i) {
        if(IsClickable(cont[i])) { return i; }
    }
    return ci;
}

int32 GetPrevClickable(iui::Widget *w, int32 ci)
{
    iui::WidgetCont &cont = w->getChildren();
    if(ci>=cont.size()) { return -1; }

    for(int32 i=ci-1; i>=0 && i<cont.size(); --i) {
        if(IsClickable(cont[i])) { return i; }
    }
    for(int32 i=cont.size()-1; i>=0; --i) {
        if(IsClickable(cont[i])) { return i; }
    }
    return ci;
}


UICursor::UICursor()
    : m_stack(), m_pos(), m_size()
{
}

void UICursor::update(iui::Float dt)
{
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
        iui::Widget *w = state.widget->getChildren()[state.index];

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

    iui::Color bg(1.0f,1.0f,1.0f,0.3f);
    iui::Rect rect(m_pos, m_size);
    iuiGetRenderer()->drawRect(rect, bg);
}

void UICursor::pushSelection(iui::Widget *v)
{
    int32 pos = GetNextClickable(v, -1);
    if(pos!=-1) {
        m_stack.push_back(State(v,pos));
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

void UICursor::setSelection( iui::Widget *v )
{
    clearSelection();
    pushSelection(v);
}

void UICursor::moveNext()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
        state.index = GetNextClickable(state.widget, state.index);
    }
}

void UICursor::movePrev()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
        state.index = GetPrevClickable(state.widget, state.index);
    }
}

void UICursor::enter()
{
    if(!m_stack.empty()) {
        State &state = m_stack.back();
        iui::Widget *w = state.widget->getChildren()[state.index];

        iui::WM_Widget wm;
        wm.type = iui::WMT_iuiOK;
        wm.from = nullptr;
        w->handleEvent(wm);
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
