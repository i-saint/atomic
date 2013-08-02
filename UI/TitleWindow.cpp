#include "stdafx.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Graphics/Renderer.h"
#include "RootWindow.h"
#include "TitleWindow.h"
#include "ConfigWindow.h"
#include "UISelector.h"
#include "Poco/DirectoryIterator.h"

namespace atm {

class StartWindow : public iui::Panel
{
public:
    StartWindow();
private:
    void onCampaign(Widget *);
    void onHorde(Widget *);
    void onEdit(Widget *);
    bool onCancel(const iui::WM_Widget &wm) override;
};

class RecordWindow : public iui::Panel
{
public:
    RecordWindow();
    void refresh();
private:
    void onSelect(Widget *);
    void onStart(Widget *);
    bool onCancel(const iui::WM_Widget &wm) override;

    iui::List   *m_li_files;
    iui::Button *m_bu_start;
    iui::String m_selection;
};


TitleWindow::TitleWindow()
    : m_start(nullptr)
    , m_record(nullptr)
{
    std::fill_n(m_buttons, _countof(m_buttons), (iui::ToggleButton*)nullptr);

    using std::placeholders::_1;
    iui::Size size(200, 25);
    float32 vspace = 55.0f;
    m_buttons[0] = iuiNew(iui::ToggleButton)(this, L"start",   iui::Rect(iui::Position(80, 400+vspace*0), size), std::bind(&TitleWindow::onStart, this, _1));
    m_buttons[1] = iuiNew(iui::ToggleButton)(this, L"record",  iui::Rect(iui::Position(80, 400+vspace*1), size), std::bind(&TitleWindow::onRecord, this, _1));
    m_buttons[2] = iuiNew(iui::ToggleButton)(this, L"config",  iui::Rect(iui::Position(80, 400+vspace*2), size), std::bind(&TitleWindow::onConfig, this, _1));
    m_buttons[3] = iuiNew(iui::ToggleButton)(this, L"exit",    iui::Rect(iui::Position(80, 400+vspace*3), size), std::bind(&TitleWindow::onExit, this, _1));

    m_start     = iuiNew(StartWindow)();
    m_record    = iuiNew(RecordWindow)();
    m_config    = iuiNew(ConfigWindow)();

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
    IFontRenderer *font = atmGetTitleFont();
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
    unselectAll();
    m_buttons[0]->setPressed(true, false);
    m_start->setVisibility(true);
    atmGetUISelector()->pushSelection(m_start);
}

void TitleWindow::onRecord(Widget *)
{
    unselectAll();
    m_buttons[1]->setPressed(true, false);
    m_record->refresh();
    m_record->setVisibility(true);
    atmGetUISelector()->pushSelection(m_record);
}

void TitleWindow::onConfig(Widget *)
{
    unselectAll();
    m_buttons[2]->setPressed(true, false);
    m_config->setVisibility(true);
    atmGetUISelector()->pushSelection(m_config);
}

void TitleWindow::onExit(Widget *)
{
    unselectAll();
    m_buttons[3]->setPressed(true, false);
    atmGetApplication()->requestExit();
}

void TitleWindow::unselectAll()
{
    Widget *windows[] = {m_start, m_record, m_config};
    for(size_t i=0; i<_countof(windows); ++i) {
        windows[i]->setVisibility(false);
    }
    for(size_t i=0; i<_countof(m_buttons); ++i) {
        m_buttons[i]->setPressed(false, false);
    }
}

void TitleWindow::setVisibility( bool v )
{
    super::setVisibility(v);
    if(v) {
        atmGetUISelector()->setSelection(this);
    }
    else {
        unselectAll();
        atmGetUISelector()->clearSelection();
    }
}

bool TitleWindow::onCancel(const iui::WM_Widget &wm)
{
    unselectAll();
    atmGetUISelector()->popSelection();
    if(atmGetUISelector()->getSelection().empty()) {
        atmGetUISelector()->setSelection(this, 0);
    }
    return true;
}


StartWindow::StartWindow()
{
    using std::placeholders::_1;
    iui::Size size(150, 25);
    float32 vspace = 40.0f;
    float n = 0.0f;
    iuiNew(iui::Button)(this, L"campaign",iui::Rect(iui::Position(0, vspace*n), size), std::bind(&StartWindow::onCampaign, this, _1));
    n += 1.0f;
    //istNew(iui::Button)(this, L"horde",   iui::Rect(iui::Position(0, vspace*n), size), std::bind(&StartWindow::onHorde, this, _1));
    //n += 1.0f;
    iuiNew(iui::Button)(this, L"edit",    iui::Rect(iui::Position(0, vspace*n), size), std::bind(&StartWindow::onEdit, this, _1));
    n += 1.0f;
}

void StartWindow::onCampaign(Widget *)
{
    GameStartConfig conf;
    atmGetApplication()->requestStartGame(conf);
    atmGetTitleWindow()->setVisibility(false);
}

void StartWindow::onHorde(Widget *)
{
    GameStartConfig conf;
    conf.gmode = GameStartConfig::GM_Horde;
    atmGetApplication()->requestStartGame(conf);
    atmGetTitleWindow()->setVisibility(false);
}

void StartWindow::onEdit( Widget * )
{
    GameStartConfig conf;
    conf.gmode = GameStartConfig::GM_Edit;
    atmGetApplication()->requestStartGame(conf);
    atmGetTitleWindow()->setVisibility(false);
}

bool StartWindow::onCancel(const iui::WM_Widget &wm)
{
    return getParent()->onCancel(wm);
}



RecordWindow::RecordWindow()
    : m_li_files(), m_bu_start()
{
    using std::placeholders::_1;
    m_li_files  = iuiNew(iui::List)(this, iui::Rect(iui::Position(0, 0), iui::Size(300.0f, 250.0f)), std::bind(&RecordWindow::onSelect, this, _1));
    m_bu_start  = iuiNew(iui::Button)(this, L"start", iui::Rect(iui::Position(0, 260.0f), iui::Size(150, 25)), std::bind(&RecordWindow::onStart, this, _1));
}

void RecordWindow::refresh()
{
    m_li_files->clearItems();

    Poco::DirectoryIterator end;
    for(Poco::DirectoryIterator it(Poco::Path("Replay")); it!=end; ++it) {
        if(it->isFile() && it->canRead()) {
            m_li_files->addListItem(ist::L(it->path()), nullptr);
        }
    }
}

void RecordWindow::onSelect( Widget *w )
{
    m_selection = static_cast<iui::List*>(w)->getSelectedItem()->getText();
}

void RecordWindow::onStart( Widget * )
{
    if(m_selection.empty()) { return; }

    GameStartConfig conf;
    conf.gmode = GameStartConfig::GM_Replay;
    std::string path = ist::S(m_selection);
    conf.path_to_replay = path;
    atmGetApplication()->requestStartGame(conf);
    atmGetTitleWindow()->setVisibility(false);
}

bool RecordWindow::onCancel(const iui::WM_Widget &wm)
{
    return getParent()->onCancel(wm);
}


} // namespace atm
