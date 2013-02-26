#include "iuiPCH.h"
#include "iuiSystem.h"
#include "iuiWidget.h"
namespace iui {


UISystem * UISystem::s_inst;

void UISystem::initializeInstance()
{
    if(s_inst==NULL) {
        s_inst = istNew(UISystem)();
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
    WMHandler   wmhandler;

    Members()
        : renderer(NULL)
        , root_widget(NULL)
        , focus(NULL)
    {
    }
};

UIRenderer* UISystem::getRenderer() const       { return m->renderer; }
Widget*     UISystem::getRootWidgets() const    { return m->root_widget; }
Widget*     UISystem::getFocus() const          { return m->focus; }

UISystem::UISystem()
{
    m->wmhandler = std::bind(&UISystem::handleWindowMessage, this, std::placeholders::_1);
    istGetAplication()->addMessageHandler(&m->wmhandler);
}

UISystem::~UISystem()
{
    if(m->root_widget) {
        m->root_widget->release();
    }
    istGetAplication()->eraseMessageHandler(&m->wmhandler);
}

bool UISystem::handleWindowMessage( const WM_Base &wm )
{
    if(m->root_widget) {
        m->root_widget->handleEvent(wm);
    }
    return false;
}

void UISystem::update( Float dt )
{
    if(m->root_widget) {
        m->root_widget->update(dt);
    }
}

void UISystem::draw()
{
    if(m->root_widget) {
        m->root_widget->draw();
    }
}

void UISystem::setFocus( Widget *v )
{
    if(m->focus) {
        // todo defocus
    }
    m->focus = v;
    if(m->focus) {
        // todo focus
    }
}


} // namespace iui
