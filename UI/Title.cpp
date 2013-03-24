#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Title.h"
#include "Poco/DirectoryIterator.h"

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

void RootWindow::update(iui::Float dt)
{
    // 動いてもマウスイベントに適切に対応できるかテスト
    //static float32 f = 0.0f;
    //f += 0.005f;
    //setPosition(iui::Position(std::cos(f)*100.0f, 0.0f));
    super::update(dt);
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


    m_start = istNew(StartWindow)();
    m_record = istNew(RecordWindow)();

    Widget *windows[] = {m_start, m_record};
    for(size_t i=0; i<_countof(windows); ++i) {
        windows[i]->setParent(this);
        windows[i]->setPosition(iui::Position(250, 300));
        windows[i]->setSize(iui::Size(500, 500));
        windows[i]->setVisibility(false);
    }
}

void TitleWindow::onStart(Widget *)
{
    GameStartConfig conf;
    atomicGetApplication()->requestStartGame(conf);
}

void TitleWindow::onRecord(Widget *)
{
    hideAll();
    m_record->setVisibility(true);
}

void TitleWindow::onConfig(Widget *)
{

}

void TitleWindow::onExit(Widget *)
{
    atomicGetApplication()->requestExit();
}

void TitleWindow::hideAll()
{
    Widget *windows[] = {m_start, m_record};
    for(size_t i=0; i<_countof(windows); ++i) {
        windows[i]->setVisibility(false);
    }
}




StartWindow::StartWindow()
{

}

void StartWindow::onCampaign(Widget *)
{

}

void StartWindow::onHorde(Widget *)
{

}

void StartWindow::onQuickJoin(Widget *)
{

}



RecordWindow::RecordWindow()
{
    using std::placeholders::_1;
    iui::List *ls  = istNew(iui::List)(this, iui::Rect(iui::Position(0, 0), iui::Size(300.0f, 250.0f)), std::bind(&RecordWindow::onSelect, this, _1));

    Poco::DirectoryIterator end;
    for(Poco::DirectoryIterator it(Poco::Path("Replay")); it!=end; ++it) {
        if(it->isFile() && it->canRead()) {
            ls->addListItem(ist::L(it->path()), NULL);
        }
    }
}

void RecordWindow::onSelect( Widget *w )
{
    GameStartConfig conf;
    conf.gmode = GameStartConfig::GM_Replay;
    std::string path = ist::S(static_cast<iui::List*>(w)->getSelectedItem()->getText());
    conf.path_to_replay = path;
    atomicGetApplication()->requestStartGame(conf);
}



ConfigWindow::ConfigWindow()
{
    using std::placeholders::_1;
    iui::Size size(150, 25);
    float32 vspace = 55.0f;
    //iui::Editbox *ed_name  = istNew(iui::Editbox)(this, atomicGetConfig()->name, iui::Rect(iui::Position(250, 300+vspace*0), size), std::bind(&ConfigWindow::onName, this, _1));
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
