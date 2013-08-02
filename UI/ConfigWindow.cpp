#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "ConfigWindow.h"

namespace atm {


ConfigWindow::ConfigWindow()
{
    using std::placeholders::_1;
    iui::Size size(150, 25);
    float32 vspace = 40.0f;

    iui::Label *lb_name     = iuiNew(iui::Label)(this, L"name", iui::Rect(iui::Position(0, 0+vspace*0), iui::Size(40, 25)));
    iui::Editbox *ed_name   = iuiNew(iui::Editbox)(this, atmGetConfig()->name, iui::Rect(iui::Position(40, 0+vspace*0), size), std::bind(&ConfigWindow::onName, this, _1));
}

void ConfigWindow::onName(Widget *w)
{
    size_t max_len = _countof(atmGetConfig()->name)-1;
    if(w->getText().size()>max_len) {
        iui::String str = w->getText();
        str.resize(max_len);
        w->setText(str);
    }
    wcscpy(atmGetConfig()->name, w->getText().c_str());
}

void ConfigWindow::onFullscreen(Widget *)
{
}

void ConfigWindow::onResolution(Widget *)
{
}

void ConfigWindow::onBGMVolume(Widget *)
{
}

void ConfigWindow::onBGMOnOff(Widget *)
{
}

void ConfigWindow::onSEVolume(Widget *)
{
}

void ConfigWindow::onSEOnOff(Widget *)
{
}

bool ConfigWindow::onCancel(const iui::WM_Widget &wm)
{
    return getParent()->onCancel(wm);
}

} // namespace atm
