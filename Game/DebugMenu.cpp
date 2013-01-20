#include "stdafx.h"
#include "DebugMenu.h"
#include "Graphics/Renderer.h"

#ifdef atomic_enable_debug_menu
namespace atomic {


DebugMenu * DebugMenu::s_inst = NULL;

void DebugMenu::initializeInstance()
{
    istAssert(s_inst==NULL, "");
    s_inst = istNew(DebugMenu)();
}

void DebugMenu::finalizeInstance()
{
    istAssert(s_inst!=NULL, "");
    istSafeDelete(s_inst);
}

DebugMenu* DebugMenu::getInstance()
{
    return s_inst;
}

DebugMenu::DebugMenu()
{
    m_root = istNew(ist::ParamNodeBase)();
}

DebugMenu::~DebugMenu()
{
    istSafeDelete(m_root);
}

void DebugMenu::update()
{

}

void DebugMenu::draw()
{
    IFontRenderer *fr = atomicGetFontRenderer();
}

ist::IParamNode* DebugMenu::getRoot()
{
    return m_root;
}

} // namespace atomic

#endif // atomic_enable_debug_menu
