#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Title.h"

namespace atm {

uint32 GetNextClickable(iui::Widget *w, uint32 i);
uint32 GetPrevClickable(iui::Widget *w, uint32 i);

bool IsClickable(iui::Widget *w)
{
    switch(w->getTypeID()) {
    case iui::WT_Button:
    case iui::WT_ToggleButton:
    case iui::WT_Checkbox:
        return true;
    default:
        return false;
    }
}

uint32 GetNextClickable(iui::Widget *w, uint32 ci)
{
    iui::WidgetCont &cont = w->getChildren();
    if(ci>=cont.size()) { return 0; }

    for(uint32 i=ci+1; i<cont.size(); ++i) {
        if(IsClickable(cont[i])) { return i; }
    }
    for(uint32 i=0; i<ci; ++i) {
        if(IsClickable(cont[i])) { return i; }
    }
    return ci;
}

uint32 GetPrevClickable(iui::Widget *w, uint32 ci)
{
    iui::WidgetCont &cont = w->getChildren();
    if(ci>=cont.size()) { return 0; }

    for(uint32 i=ci-1; i>=0 && i<cont.size(); --i) {
        if(IsClickable(cont[i])) { return i; }
    }
    for(uint32 i=cont.size()-1; i>=0; --i) {
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
        if(!state.widget->isVisibleAbs()) {
            return;
        }
    }

    iui::Color bg(1.0f,1.0f,1.0f,0.3f);
    iui::Rect rect(m_pos, m_size);
    iuiGetRenderer()->drawRect(rect, bg);
}

void UICursor::pushStack(iui::Widget *v)
{
    m_stack.push_back(State(v,0));
}

void UICursor::popStack()
{
    if(!m_stack.empty()) {
        m_stack.pop_back();
    }
}

void UICursor::clearStack()
{
    m_stack.clear();
}

void UICursor::setTarget( iui::Widget *v )
{
    clearStack();
    pushStack(v);
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

        iui::WM_Mouse wm;
        wm.type = iui::WMT_MouseDown;
        wm.mouse_pos = w->getPositionAbs() + (w->getSize()*0.5f);
        wm.button.left = 1;
        w->handleEvent(wm);
        wm.type = iui::WMT_MouseUp;
        w->handleEvent(wm);
    }
}

void UICursor::cancel()
{
    popStack();
}


} // namespace atm
