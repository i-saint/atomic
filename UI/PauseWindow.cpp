#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "PauseWindow.h"
#include "ConfigWindow.h"
#include "RootWindow.h"
#include "UISelector.h"

namespace atm {

PauseWindow::PauseWindow()
    : m_config(nullptr)
{
    super::setVisibility(false);

    using std::placeholders::_1;
    iui::Size size(200, 25);
    float32 vspace = 55.0f;
    m_buttons[0] = iuiNew(iui::ToggleButton)(this, L"resume",  iui::Rect(iui::Position(80, 400+vspace*0), size), std::bind(&PauseWindow::onResume, this, _1));
    m_buttons[1] = iuiNew(iui::ToggleButton)(this, L"config",  iui::Rect(iui::Position(80, 400+vspace*1), size), std::bind(&PauseWindow::onConfig, this, _1));
    m_buttons[2] = iuiNew(iui::ToggleButton)(this, L"title",   iui::Rect(iui::Position(80, 400+vspace*2), size), std::bind(&PauseWindow::onTitle, this, _1));
    m_buttons[3] = iuiNew(iui::ToggleButton)(this, L"exit",    iui::Rect(iui::Position(80, 400+vspace*3), size), std::bind(&PauseWindow::onExit, this, _1));

    m_config    = iuiNew(ConfigWindow)();

    Widget *windows[] = {m_config};
    for(size_t i=0; i<_countof(windows); ++i) {
        windows[i]->setParent(this);
        windows[i]->setPosition(iui::Position(350, 400));
        windows[i]->setSize(iui::Size(500, 500));
        windows[i]->setVisibility(false);
    }
}

void PauseWindow::setVisibility( bool v, bool e )
{
    super::setVisibility(v,e);
    if(v) {
        atmPause(true);
        atmGetUISelector()->setSelection(this);
    }
    else {
        atmPause(false);
        atmGetUISelector()->clearSelection();
        unselectAll();
    }
}

void PauseWindow::unselectAll()
{
    Widget *windows[] = {m_config};
    for(size_t i=0; i<_countof(windows); ++i) {
        windows[i]->setVisibility(false);
    }
    for(size_t i=0; i<_countof(m_buttons); ++i) {
        m_buttons[i]->setPressed(false, false);
    }
}

bool PauseWindow::onCancel( const iui::WM_Widget &wm )
{
    onResume(nullptr);
    return true;
}

void PauseWindow::onResume( Widget * )
{
    unselectAll();
    m_buttons[0]->setPressed(true, false);
    setVisibility(false);
}

void PauseWindow::onConfig( Widget * )
{
    unselectAll();
    m_buttons[1]->setPressed(true, false);
    m_config->setVisibility(true);
    atmGetUISelector()->pushSelection(m_config);
}

void PauseWindow::onTitle( Widget * )
{
    unselectAll();
    m_buttons[2]->setPressed(true, false);
    atmRequestReturnToTitleScreen();
    setVisibility(false);
    atmPause(false);
}

void PauseWindow::onExit( Widget * )
{
    unselectAll();
    m_buttons[3]->setPressed(true, false);
    atmRequestExit();
    setVisibility(false);
    atmPause(false);
}

void PauseWindow::drawCallback()
{

}

} // namespace atm
