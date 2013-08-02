#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "RootWindow.h"
#include "TitleWindow.h"
#include "PauseWindow.h"
#include "ConfigWindow.h"
#include "LogWindow.h"
#include "UISelector.h"

namespace atm {

RootWindow::RootWindow()
    : m_title(nullptr)
    , m_pause(nullptr)
    , m_log(nullptr)
    , m_cursor(nullptr)
{
    setSize(iui::Size(atmGetWindowSize().x, atmGetWindowSize().y));
    m_title  = iuiNew(TitleWindow)();
    m_pause  = iuiNew(PauseWindow)();
    m_log    = iuiNew(LogWindow)();
    m_cursor = istNew(UISelector)();
    m_cursor->setSelection(m_title);

    Widget *widgets[] = {m_title, m_pause, m_log};
    for(size_t i=0; i<_countof(widgets); ++i) {
        widgets[i]->setParent(this);
        widgets[i]->setSize(getSize());
    }
}

RootWindow::~RootWindow()
{
    istSafeDelete(m_cursor);
}

void RootWindow::update(iui::Float dt)
{
    // 動いてもマウスイベントに適切に対応できるかテスト
    //static float32 f = 0.0f;
    //f += 0.005f;
    //setPosition(iui::Position(std::cos(f)*100.0f, 0.0f));
    super::update(dt);
    m_cursor->update(dt);
}

void RootWindow::draw()
{
    super::draw();
    {
        const iui::Position &pos = iui::Position();
        const iui::Size &size = iuiGetSystem()->getScreen().getSize();
        const iui::Rect &screen = iuiGetSystem()->getScreen();
        const iui::Size viewport = iui::Size(istGetAplication()->getWindowSize().x, istGetAplication()->getWindowSize().y);
        iui::Size r = viewport/screen.size;
        iuiGetRenderer()->setViewport( (int32)(pos.x*r.x-0.5f), (int32)(viewport.y-(pos.y+size.y)*r.y-0.5f), (int32)(size.x*r.x+1.0f), (int32)(size.y*r.y+1.0f) );
        iuiGetRenderer()->setScreen(-0.5f, -0.5f, size.x+1.0f, size.y+1.0f);

        m_cursor->draw();
    }
}

iui::RootWindow*    atmCreateRootWindow()   { return iuiNew(RootWindow)(); }
iui::RootWindow*    atmGetRootWindow()      { return iuiGetRootWindow(); }
iui::Widget*        atmGetTitleWindow()     { return ((RootWindow*)atmGetRootWindow())->getTitleWindow(); }
iui::Widget*        atmGetPauseWindow()     { return ((RootWindow*)atmGetRootWindow())->getPauseWindow(); }
iui::Widget*        atmGetLogWindow()       { return ((RootWindow*)atmGetRootWindow())->getLogWindow(); }
UISelector*           atmGetUISelector()      { return ((RootWindow*)atmGetRootWindow())->getCursor(); }

void atmPauseAndShowPauseMenu()
{
    atmGetPauseWindow()->setVisibility(true);
}

} // namespace atm
