#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
#include "iuiWindow.h"
#include "iuiRenderer.h"
namespace iui {


UISystem * UISystem::s_inst;

void UISystem::initializeInstance()
{
    if(s_inst==NULL) {
        istNew(UISystem)();
    }
}

void UISystem::finalizeInstance()
{
    istSafeDelete(s_inst);
}

UISystem* UISystem::getInstance()
{
    return s_inst;
}


struct UISystem::Members
{
    typedef UISystem::WMHandler WMHandler;

    UIRenderer  *renderer;
    Widget      *root_widget;
    Widget      *focus;
    WidgetCont  new_widgets;
    WMHandler   wmhandler;
    Rect        screen;

    Members()
        : renderer(NULL)
        , root_widget(NULL)
        , focus(NULL)
        , screen()
    {
    }
};
istMemberPtrImpl(UISystem,Members)

UIRenderer* UISystem::getRenderer() const       { return m->renderer; }
Widget*     UISystem::getRootWindow() const    { return m->root_widget; }
Widget*     UISystem::getFocus() const          { return m->focus; }
const Rect& UISystem::getScreen() const         { return m->screen; }

void UISystem::setRootWindow( Widget *root )
{
    m->root_widget = root;
}

void UISystem::setScreen( float32 width, float32 height )
{
    m->screen = Rect(Position(), Size(width, height));
    if(m->root_widget) {
        m->root_widget->setPosition(m->screen.pos);
        m->root_widget->setSize(m->screen.size);
    }
}


UISystem::UISystem()
{
    s_inst = this;

    m->renderer = CreateUIRenderer();
    m->wmhandler = std::bind(&UISystem::handleWindowMessage, this, std::placeholders::_1);

    ist::Application *app = istGetAplication();
    uvec2 wsize = app->getWindowSize();
    app->addMessageHandler(&m->wmhandler);
    setScreen((float32)wsize.x, (float32)wsize.y);
}

UISystem::~UISystem()
{
    istGetAplication()->eraseMessageHandler(&m->wmhandler);
    istSafeRelease(m->renderer);
    istSafeRelease(m->root_widget);

    s_inst = NULL;
}


bool UISystem::handleWindowMessage( const ist::WM_Base &wm )
{
    if(m->root_widget) {
        handleWindowMessageR(m->root_widget, wm);
    }
    switch(wm.type) {
    case WMT_WidgetDelete:
        {
            const WM_Widget &mes = reinterpret_cast<const WM_Widget&>(wm);
            if(m->focus==mes.from) {
                m->focus->setFocus(false);
                m->focus = NULL;
            }
        }
        break;
    }
    return false;
}

void UISystem::sendMessage( const WM_Base &wm )
{
    handleWindowMessage(wm);
}

bool UISystem::handleWindowMessageR( Widget *widget, const WM_Base &wm )
{
    // 子が先
    bool handled = false;
    widget->eachChildren([&](Widget *c){
        if(!handled && handleWindowMessageR(c, wm)) { handled=true; }
    });
    if(!handled && widget->handleEvent(wm)) { handled=true; }
    return handled;
}


void UISystem::update( Float dt )
{
    std::for_each(m->new_widgets.begin(), m->new_widgets.end(), [&](Widget *w){
        WM_Widget mes;
        mes.type = WMT_WidgetCretate;
        mes.from = w;
        sendMessage(mes);
    });
    m->new_widgets.clear();

    updateR(m->root_widget, dt);
}

void UISystem::updateR( Widget *widget, Float dt )
{
    // 子が先
    widget->eachChildren([&](Widget *c){ updateR(c, dt); });
    widget->update(dt);
}


void UISystem::draw()
{
    m->renderer->begin();
    m->renderer->setScreen(getScreen().getSize().x, getScreen().getSize().y);
    drawR(m->root_widget);
    m->renderer->end();
}

void UISystem::drawR( Widget *widget )
{
    // 親が先 (奥→手前の順)
    widget->draw();
    widget->eachChildrenReverse([&](Widget *c){ drawR(c); });
}


void UISystem::setFocus( Widget *v )
{
    m->focus = v;
}

void UISystem::notifyNewWidget( Widget *w )
{
    m->new_widgets.insert(w);
}


} // namespace iui
