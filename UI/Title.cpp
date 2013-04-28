#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Graphics/Renderer.h"
#include "Title.h"
#include "Poco/DirectoryIterator.h"

namespace atomic {



RootWindow::RootWindow()
    : m_title(NULL)
    , m_log(NULL)
{
    setSize(iui::Size(atomicGetWindowSize().x, atomicGetWindowSize().y));

    m_title  = istNew(TitleWindow)();
    m_log    = istNew(LogWindow)();

    Widget *widgets[] = {m_title, m_log};
    for(size_t i=0; i<_countof(widgets); ++i) {
        widgets[i]->setParent(this);
        widgets[i]->setSize(getSize());
    }
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
    std::fill_n(m_buttons, _countof(m_buttons), (iui::ToggleButton*)NULL);

    using std::placeholders::_1;
    iui::Size size(200, 25);
    float32 vspace = 55.0f;
    m_buttons[0] = istNew(iui::ToggleButton)(this, L"start",   iui::Rect(iui::Position(80, 400+vspace*0), size), std::bind(&TitleWindow::onStart, this, _1));
    m_buttons[1] = istNew(iui::ToggleButton)(this, L"record",  iui::Rect(iui::Position(80, 400+vspace*1), size), std::bind(&TitleWindow::onRecord, this, _1));
    m_buttons[2] = istNew(iui::ToggleButton)(this, L"config",  iui::Rect(iui::Position(80, 400+vspace*2), size), std::bind(&TitleWindow::onConfig, this, _1));
    m_buttons[3] = istNew(iui::ToggleButton)(this, L"exit",    iui::Rect(iui::Position(80, 400+vspace*3), size), std::bind(&TitleWindow::onExit, this, _1));


    m_start     = istNew(StartWindow)();
    m_record    = istNew(RecordWindow)();
    m_config    = istNew(ConfigWindow)();

    Widget *windows[] = {m_start, m_record, m_config};
    for(size_t i=0; i<_countof(windows); ++i) {
        windows[i]->setParent(this);
        windows[i]->setPosition(iui::Position(350, 400));
        windows[i]->setSize(iui::Size(500, 500));
        windows[i]->setVisibility(false);
    }
}

void TitleWindow::draw()
{
    IFontRenderer *font = atomicGetTitleFont();
    const iui::Size &size = iuiGetRootWindow()->getSize();
    font->setScreen(0.0f, size.x, size.y, 0.0f);
    font->setSize(120.0f);
    font->setSpacing(5.0f);
    font->setColor(vec4(1.0f, 1.0f, 1.0f, 0.8f));
    font->addText(vec2(100.0f, 150.0f), L"atomic");
    font->draw();
}

void TitleWindow::onStart(Widget *)
{
    hideAll();
    m_buttons[0]->setPressed(true, false);
    m_start->setVisibility(true);
}

void TitleWindow::onRecord(Widget *)
{
    hideAll();
    m_buttons[1]->setPressed(true, false);
    m_record->setVisibility(true);
}

void TitleWindow::onConfig(Widget *)
{
    hideAll();
    m_buttons[2]->setPressed(true, false);
    m_config->setVisibility(true);
}

void TitleWindow::onExit(Widget *)
{
    hideAll();
    m_buttons[3]->setPressed(true, false);
    atomicGetApplication()->requestExit();
}

void TitleWindow::hideAll()
{
    Widget *windows[] = {m_start, m_record, m_config};
    for(size_t i=0; i<_countof(windows); ++i) {
        windows[i]->setVisibility(false);
    }
    for(size_t i=0; i<_countof(m_buttons); ++i) {
        m_buttons[i]->setPressed(false, false);
    }
}




StartWindow::StartWindow()
{
    using std::placeholders::_1;
    iui::Size size(150, 25);
    float32 vspace = 40.0f;
    iui::Button *bu_start   = istNew(iui::Button)(this, L"campaign",    iui::Rect(iui::Position(0, vspace*0), size), std::bind(&StartWindow::onCampaign, this, _1));
    iui::Button *bu_record  = istNew(iui::Button)(this, L"horde",       iui::Rect(iui::Position(0, vspace*1), size), std::bind(&StartWindow::onHorde, this, _1));
    iui::Button *bu_qjoin   = istNew(iui::Button)(this, L"quick join",  iui::Rect(iui::Position(0, vspace*2), size), std::bind(&StartWindow::onQuickJoin, this, _1));
}

void StartWindow::onCampaign(Widget *)
{
    GameStartConfig conf;
    atomicGetApplication()->requestStartGame(conf);
    getParent()->setVisibility(false);
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
    float32 vspace = 40.0f;

    iui::Label *lb_name = istNew(iui::Label)(this, L"name", iui::Rect(iui::Position(0, 0+vspace*0), iui::Size(40, 25)));
    iui::Editbox *ed_name  = istNew(iui::Editbox)(this, atomicGetConfig()->name, iui::Rect(iui::Position(40, 0+vspace*0), size), std::bind(&ConfigWindow::onName, this, _1));
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


LogWindow::LogWindow()
{

}

} // namespace atomic
