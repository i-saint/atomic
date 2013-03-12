#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Title.h"

namespace atomic {



RootWindow::RootWindow()
    : m_title(NULL)
    , m_config(NULL)
    , m_log(NULL)
{
    m_title  = istNew(TitleWindow)();
    m_title->setParent(this);

    m_config = istNew(ConfigWindow)();
    m_config->setParent(this);

    m_log    = istNew(LogWindow)();
    m_log->setParent(this);
}


TitleWindow::TitleWindow()
    : m_start(NULL)
    , m_record(NULL)
{
    using std::placeholders::_1;
    iui::Size size(150, 25);
    float32 vspace = 55.0f;
    iui::Button *bu_start   = istNew(iui::Button)(this, L"start",   iui::Rect(iui::Position(80, 300+vspace*0), size), std::bind(&TitleWindow::onStart, this, _1));
    iui::Button *bu_record  = istNew(iui::Button)(this, L"record",  iui::Rect(iui::Position(80, 300+vspace*1), size), std::bind(&TitleWindow::onRecord, this, _1));
    iui::Button *bu_config  = istNew(iui::Button)(this, L"config",  iui::Rect(iui::Position(80, 300+vspace*2), size), std::bind(&TitleWindow::onConfig, this, _1));
    iui::Button *bu_exit    = istNew(iui::Button)(this, L"exit",    iui::Rect(iui::Position(80, 300+vspace*3), size), std::bind(&TitleWindow::onExit, this, _1));
}

void TitleWindow::onStart(Widget *)
{
    setVisibility(false);
}

void TitleWindow::onRecord(Widget *)
{

}

void TitleWindow::onConfig(Widget *)
{

}

void TitleWindow::onExit(Widget *)
{
    atomicGetApplication()->requestExit();
}


ConfigWindow::ConfigWindow()
{
    using std::placeholders::_1;
    iui::Size size(150, 25);
    float32 vspace = 55.0f;
    iui::Editbox *ed_name  = istNew(iui::Editbox)(this, atomicGetConfig()->name, iui::Rect(iui::Position(250, 300+vspace*0), size), std::bind(&ConfigWindow::onName, this, _1));
    //iui::Button *bu_record  = istNew(iui::Button)(this, L"record",  iui::Rect(iui::Position(80, 300+vspace*1), size), std::bind(&TitleWindow::onRecord, this, _1));
    //iui::Button *bu_config  = istNew(iui::Button)(this, L"config",  iui::Rect(iui::Position(80, 300+vspace*2), size), std::bind(&TitleWindow::onConfig, this, _1));
    //iui::Button *bu_exit    = istNew(iui::Button)(this, L"exit",    iui::Rect(iui::Position(80, 300+vspace*3), size), std::bind(&TitleWindow::onExit, this, _1));
}

void ConfigWindow::onName(Widget *w)
{
    size_t max_len = _countof(atomicGetConfig()->name)-1;
    if(w->getText().size()>max_len) {
        iui::String str = w->getText();
        str.resize(max_len);
        w->setText(str);
    }
    wcscpy(atomicGetConfig()->name, w->getText().c_str());
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


LogWindow::LogWindow()
{

}

} // namespace atomic
