#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Title.h"

namespace atomic {


class TitleWindow : public iui::Panel
{
public:
    TitleWindow();

private:
    void onStart(Widget *);
    void onRecord(Widget *);
    void onConfig(Widget *);
    void onCredit(Widget *);
    void onExit(Widget *);

    StartWindow     *m_start;
    RecordWindow    *m_record;
};

class ConfigWindow : public iui::Panel
{
public:
    ConfigWindow();
private:
    void onResolution();
};

class LogWindow : public iui::Panel
{
public:
    LogWindow();
private:
    void onTextEnter(Widget *);
};


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
    iui::Button *bu_credit  = istNew(iui::Button)(this, L"credit",  iui::Rect(iui::Position(80, 300+vspace*3), size), std::bind(&TitleWindow::onCredit, this, _1));
    iui::Button *bu_exit    = istNew(iui::Button)(this, L"exit",    iui::Rect(iui::Position(80, 300+vspace*4), size), std::bind(&TitleWindow::onExit, this, _1));
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

void TitleWindow::onCredit(Widget *)
{

}

void TitleWindow::onExit(Widget *)
{
    atomicGetApplication()->requestExit();
}

ConfigWindow::ConfigWindow()
{
}

LogWindow::LogWindow()
{

}

} // namespace atomic
