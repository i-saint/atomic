#include "atmPCH.h"
#include "Engine/Game/AtomicApplication.h"
#include "Engine/Game/AtomicGame.h"
#include "Util.h"
#include "ConfigWindow.h"
#include "RootWindow.h"
#include "UISelector.h"

namespace atm {

void EnumerateAvailableResolutiuons(const std::function<void (const uvec2 &)> &f)
{
    std::vector<uvec2> resolutions;
#ifdef WIN32
    ::DEVMODE dm;
    for(int i=0; ::EnumDisplaySettings(0, i, &dm); ++i) {
        if(dm.dmBitsPerPel>16) {
            resolutions.push_back( uvec2(dm.dmPelsWidth, dm.dmPelsHeight) );
        }
    }
    std::sort(resolutions.begin(), resolutions.end(), [](const uvec2 &a, const uvec2 &b){
        return a.x==b.x ? a.y<b.y : a.x<b.x;
    });
    resolutions.erase(std::unique(resolutions.begin(), resolutions.end()), resolutions.end());
#else // WIN32 
    resolutions.push_back(uvec2(640, 480));
    resolutions.push_back(uvec2(800, 600));
    resolutions.push_back(uvec2(1024, 768));
    resolutions.push_back(uvec2(1280, 800));
    resolutions.push_back(uvec2(1280, 960));
    resolutions.push_back(uvec2(1280, 1024));
#endif // WIN32 
    std::for_each(resolutions.begin(), resolutions.end(), f);
}

ConfigWindow::ConfigWindow()
{
    setVisibility(false);

    using std::placeholders::_1;
    float32 vspace = 30.0f;
    float32 hspace = 270;
    float32 labelw = 100.0f;
    iui::Size size(150, 25);
    iui::Position pos;

                iuiNew(iui::Label)(this, L"name", iui::Rect(pos, iui::Size(labelw, 25)));
    m_bu_name = iuiNew(iui::ToggleButton)(this, L"", iui::Rect(pos+iui::Position(labelw, 0), size), std::bind(&ConfigWindow::onNameW, this, _1));
    m_vw_name = iuiNew(EditboxWindow)(this, L"", iui::Rect(pos+iui::Position(hspace, 0), iui::Size(400, 30)), std::bind(&ConfigWindow::onNameV, this, _1));
    m_buttons.push_back(m_bu_name);
    m_windows.push_back(m_vw_name);

    pos += iui::Position(0.0f, vspace);

                iuiNew(iui::Label)(this, L"resolution", iui::Rect(pos, iui::Size(labelw, 25)));
    m_bu_reso = iuiNew(iui::ToggleButton)(this, L"", iui::Rect(pos+iui::Position(labelw, 0), size), std::bind(&ConfigWindow::onResolutionW, this, _1));
    m_vw_reso = iuiNew(ListWindow)(this, iui::Rect(pos+iui::Position(hspace, 0), iui::Size(300, 180)), std::bind(&ConfigWindow::onResolutionV, this, _1));
    m_buttons.push_back(m_bu_reso);
    m_windows.push_back(m_vw_reso);

    pos += iui::Position(0.0f, vspace);
    
                iuiNew(iui::Label)(this, L"httpd port", iui::Rect(pos, iui::Size(labelw, 25)));
    m_bu_http = iuiNew(iui::ToggleButton)(this, L"", iui::Rect(pos+iui::Position(labelw, 0), size), std::bind(&ConfigWindow::onPortW, this, _1));
    m_vw_http = iuiNew(EditboxWindow)(this, L"", iui::Rect(pos+iui::Position(hspace, 0), iui::Size(300, 180)), std::bind(&ConfigWindow::onPortV, this, _1));
    m_buttons.push_back(m_bu_http);
    m_windows.push_back(m_vw_http);

    pos += iui::Position(0.0f, vspace);
        
                  iuiNew(iui::Label)(this, L"graphics", iui::Rect(pos, iui::Size(labelw, 25)));
    m_bu_glevel = iuiNew(iui::ToggleButton)(this, L"", iui::Rect(pos+iui::Position(labelw, 0), size), std::bind(&ConfigWindow::onRenderW, this, _1));
    m_vw_glevel = iuiNew(ListWindow)(this, iui::Rect(pos+iui::Position(hspace, 0), iui::Size(300, 180)), std::bind(&ConfigWindow::onRenderV, this, _1));
    m_buttons.push_back(m_bu_glevel);
    m_windows.push_back(m_vw_glevel);

    pos += iui::Position(0.0f, vspace);

    m_lb_desc = iuiNew(iui::Label)(this, L"", iui::Rect(pos, iui::Size(600.0f, 300.0f)));

    each(m_buttons, [](iui::ToggleButton *bu){
        bu->getStyle()->setTextHSpacing(0.75f);
    });

    EnumerateAvailableResolutiuons([&](const uvec2 &res){
        wchar_t tmp[64];
        istSPrintf(tmp, L"%dx%d", res.x, res.y);
        m_vw_reso->getList()->addListItem(tmp, nullptr);
    });

    {
        typedef std::tuple<const wchar_t*, size_t> pair;
        pair graphics_options[] = {
            pair(L"low",     atmE_Graphics_Low   ),
            pair(L"medium",  atmE_Graphics_Medium),
            pair(L"high",    atmE_Graphics_High  ),
            //pair(L"custom",  atmE_Graphics_Custom),
        };
        each(graphics_options, [&](const pair &p){
            m_vw_glevel->getList()->addListItem(std::get<0>(p), (void*)std::get<1>(p));
        });
    }

    hideAll();
    sync();
}

void ConfigWindow::setVisibility( bool v, bool e )
{
    super::setVisibility(v,e);
    if(v) {
        sync();
    }
}

bool ConfigWindow::onCancel(const iui::WM_Widget &wm)
{
    if(atmGetUISelector()->popSelection(this)) {
        hideAll();
    }
    else {
        return getParent()->onCancel(wm);
    }
    return true;
}

void ConfigWindow::sync()
{
    AtomicConfig &conf = *atmGetConfig();
    wchar_t tmp[128];

    m_bu_name->setText(atmGetConfig()->name);
    m_vw_name->getEdit()->setText(conf.name, false);

    istSPrintf(tmp, L"%dx%d", conf.window_size.x, conf.window_size.y);
    m_bu_reso->setText(tmp);

    istSPrintf(tmp, L"%d", conf.leveleditor_port);
    m_bu_http->setText(tmp);
    m_vw_http->getEdit()->setText(tmp, false);

    switch(conf.graphics_level) {
    case atmE_Graphics_Low:    m_bu_glevel->setText(L"low");    break;
    case atmE_Graphics_Medium: m_bu_glevel->setText(L"medium"); break;
    case atmE_Graphics_High:   m_bu_glevel->setText(L"high");   break;
    case atmE_Graphics_Custom: m_bu_glevel->setText(L"custom"); break;
    }
}

void ConfigWindow::hideAll()
{
    each(m_windows, [](iui::Panel *p){ p->setVisibility(false); });
    each(m_buttons, [](iui::ToggleButton *p){ p->setPressed(false, false); });
}

void ConfigWindow::onNameW(Widget *w)
{
    hideAll();
    m_bu_name->setPressed(true, false);
    m_vw_name->setVisibility(true);
    atmGetUISelector()->pushSelection(m_vw_name);
}

void ConfigWindow::onResolutionW(Widget *)
{
    hideAll();
    m_bu_reso->setPressed(true, false);
    m_vw_reso->setVisibility(true);
    atmGetUISelector()->pushSelection(m_vw_reso);
}

void ConfigWindow::onPortW(Widget *)
{
    hideAll();
    m_bu_http->setPressed(true, false);
    m_vw_http->setVisibility(true);
    atmGetUISelector()->pushSelection(m_vw_http);
}

void ConfigWindow::onRenderW(Widget *)
{
    hideAll();
    m_bu_glevel->setPressed(true, false);
    m_vw_glevel->setVisibility(true);
    atmGetUISelector()->pushSelection(m_vw_glevel);
}


void ConfigWindow::onBGMW(Widget *)
{
}

void ConfigWindow::onSEW(Widget *)
{
}



void ConfigWindow::onNameV(Widget *w)
{
    size_t max_len = _countof(atmGetConfig()->name)-1;
    if(w->getText().size()>max_len) {
        iui::String str = w->getText();
        str.resize(max_len);
        w->setText(str);
    }
    wcscpy(atmGetConfig()->name, w->getText().c_str());
    hideAll();
    sync();
    atmGetUISelector()->popSelection(this);
}

void ConfigWindow::onResolutionV(Widget *w)
{
    auto *ls = static_cast<iui::List*>(w);
    if(const iui::ListItem *item = ls->getSelectedItem()) {
        ivec2 res;
        if(swscanf(item->getText().c_str(), L"%dx%d", &res.x,&res.y)==2) {
            atmGetConfig()->window_size = res;
        }
    }
    hideAll();
    sync();
    atmGetUISelector()->popSelection(this);
}

void ConfigWindow::onPortV(Widget *w)
{
    uint32 tmp = 0;
    if(swscanf(w->getText().c_str(), L"%d", &tmp)==0) {
        atmGetConfig()->leveleditor_port = tmp;
    }
    hideAll();
    sync();
    atmGetUISelector()->popSelection(this);
}

void ConfigWindow::onRenderV(Widget *w)
{
    auto *ls = static_cast<iui::List*>(w);
    if(const iui::ListItem *item = ls->getSelectedItem()) {
        AtomicConfig &conf = *atmGetConfig();
        size_t l = (size_t)item->getUserData();
        switch(l) {
        case atmE_Graphics_Low:
            conf.lighting_level = atmE_Lighting_Low;
            conf.bg_level = atmE_BGResolution_x4;
            conf.posteffect_bloom = false;
            conf.show_bloodstain = false;
            break;
        case atmE_Graphics_Medium:
            conf.lighting_level = atmE_Lighting_Medium;
            conf.bg_level = atmE_BGResolution_x2;
            conf.posteffect_bloom = true;
            conf.show_bloodstain = false;
            break;
        case atmE_Graphics_High:
            conf.lighting_level = atmE_Lighting_High;
            conf.bg_level = atmE_BGResolution_x1;
            conf.posteffect_bloom = true;
            conf.show_bloodstain = true;
            break;
        case atmE_Graphics_Custom:
            break;
        }
        conf.graphics_level = l;
    }
    hideAll();
    sync();
    atmGetUISelector()->popSelection(this);
}

void ConfigWindow::onBGMV(Widget *)
{
}

void ConfigWindow::onSEV(Widget *)
{
}

} // namespace atm
