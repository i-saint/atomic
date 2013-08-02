#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
#include "iuiWindow.h"
#include "iuiRenderer.h"
#include "iuiUtilities.h"
namespace iui {


UISystem * UISystem::s_inst;

void UISystem::initializeInstance()
{
    if(s_inst==NULL) {
        iuiNew(UISystem)();
    }
}

void UISystem::finalizeInstance()
{
    iuiSafeDelete(s_inst);
}

UISystem* UISystem::getInstance()
{
    return s_inst;
}


UIRenderer*     UISystem::getRenderer() const   { return m_renderer; }
RootWindow*     UISystem::getRootWindow() const { return m_root; }
Widget*         UISystem::getFocus() const      { return m_focus; }
const Position& UISystem::getMousePos() const   { return m_mouse_pos;}
const Rect&     UISystem::getScreen() const     { return m_screen; }

void UISystem::setRootWindow( RootWindow *root )
{
    m_root = root;
    if(m_root) {
        m_root->setPosition(m_screen.pos);
        m_root->setSize(m_screen.size);
    }
}

void UISystem::setScreen( float32 width, float32 height )
{
    m_screen = Rect(Position(), Size(width, height));
}


UISystem::UISystem()
    : m_renderer(nullptr)
    , m_root(nullptr)
    , m_focus(nullptr)
    , m_screen()
{
    s_inst = this;

    m_renderer = CreateUIRenderer();
    m_wmhandler = std::bind(&UISystem::handleWindowMessage, this, std::placeholders::_1);

    ist::Application *app = istGetAplication();
    uvec2 wsize = app->getWindowSize();
    app->addMessageHandler(&m_wmhandler);
    setScreen((float32)wsize.x, (float32)wsize.y);
}

UISystem::~UISystem()
{
    istGetAplication()->eraseMessageHandler(&m_wmhandler);
    istSafeRelease(m_renderer);
    istSafeRelease(m_root);

    s_inst = NULL;
}


bool UISystem::handleWindowMessage( const ist::WM_Base &wm )
{
    const ist::WM_Base *pwm = &wm;
    WM_Mouse mouse;

    switch(wm.type) {
    case WMT_MouseDown:
    case WMT_MouseUp:
    case WMT_MouseMove:
        {
            mouse = WM_Mouse::cast(wm);
            pwm = &mouse;
            uvec2 wsize = istGetAplication()->getWindowSize();
            vec2 r = vec2(wsize.x, wsize.y) / m_screen.size;
            mouse.mouse_pos /= r;
            m_mouse_pos = mouse.mouse_pos;
        }
    case WMT_iuiDelete:
        {
            auto &mes = WM_Widget::cast(wm);
            if(mes.from!=NULL && m_focus==mes.from) {
                m_focus->setFocus(false);
                m_focus = NULL;
            }
        }
        break;
    }
    if(m_root) {
        handleWindowMessageR(m_root, *pwm);
    }
    return false;
}

void UISystem::sendMessage( const WM_Base &wm )
{
    handleWindowMessage(wm);
}

bool UISystem::handleWindowMessageR( Widget *widget, const WM_Base &wm )
{
    if(!widget->isVisible()) { return false; }
    // 子が先
    bool handled = false;
    widget->eachChildren([&](Widget *c){
        if(!handled) {
            if(handleWindowMessageR(c, wm)) {
                handled = true;
            }
        }
    });
    if(!handled) {
        if(widget->handleEvent(wm)) { handled=true; }
    }
    return handled;
}


void UISystem::update( Float dt )
{
    std::for_each(m_new_widgets.begin(), m_new_widgets.end(), [&](Widget *w){
        WM_Widget mes;
        mes.type = WMT_iuiCreate;
        mes.from = w;
        sendMessage(mes);
    });
    m_new_widgets.clear();

    updateR(m_root, dt);
}

void UISystem::updateR( Widget *widget, Float dt )
{
    // 子が先
    widget->eachChildren([&](Widget *c){ updateR(c, dt); });
    widget->update(dt);
}


void UISystem::draw()
{
    m_renderer->begin();
    drawR(m_root);
    m_renderer->setScreen(getScreen().getSize().x, getScreen().getSize().y);
    m_renderer->setViewport(0,0);
    m_renderer->end();
}

void UISystem::drawR( Widget *widget )
{
    // 親が先 (奥→手前の順)
    if(widget->isVisible()) {
        const Position &pos = widget->getPositionAbs();
        const Size &size = widget->getSize();
        const Rect &screen = iuiGetSystem()->getScreen();
        const Size viewport = Size(istGetAplication()->getWindowSize().x, istGetAplication()->getWindowSize().y);
        Size r = viewport/screen.size;
        iuiGetRenderer()->setViewport( (int32)(pos.x*r.x-0.5f), (int32)(viewport.y-(pos.y+size.y)*r.y-0.5f), (int32)(size.x*r.x+1.0f), (int32)(size.y*r.y+1.0f) );
        iuiGetRenderer()->setScreen(-0.5f, -0.5f, size.x+1.0f, size.y+1.0f);

        widget->draw();
        widget->eachChildrenReverse([&](Widget *c){ drawR(c); });
    }
}


void UISystem::setFocus( Widget *v )
{
    if(m_focus!=v) {
        if(m_focus) {
            WM_Widget wm;
            wm.type = WMT_iuiLostFocus;
            wm.from = m_focus;
            sendMessage(wm);
        }
        m_focus = v;
        if(v) {
            WM_Widget wm;
            wm.type = WMT_iuiGainFocus;
            wm.from = v;
            sendMessage(wm);
        }
    }
}

void UISystem::notifyNewWidget( Widget *w )
{
    m_new_widgets.insert(w);
}


} // namespace iui
