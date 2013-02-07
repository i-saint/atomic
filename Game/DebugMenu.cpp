#include "stdafx.h"
#include "DebugMenu.h"
#include "Game/AtomicApplication.h"
#include "Graphics/Renderer.h"

#ifdef atomic_enable_debug_menu
namespace atomic {


DebugMenu * DebugMenu::s_inst = NULL;

void DebugMenu::initializeInstance()
{
    istAssert(s_inst==NULL);
    s_inst = istNew(DebugMenu)();
}

void DebugMenu::finalizeInstance()
{
    istAssert(s_inst!=NULL);
    istSafeDelete(s_inst);
}

DebugMenu* DebugMenu::getInstance()
{
    return s_inst;
}

DebugMenu::DebugMenu()
    : m_root(NULL)
    , m_active(false)
{
    m_root = istNew(ist::ParamNodeBase)();
    m_root->setOpened(true);
}

DebugMenu::~DebugMenu()
{
    istSafeDelete(m_root);
}

void DebugMenu::update()
{
    const ist::KeyboardState &kb = atomicGetApplication()->getKeyboardState();
    if(kb.isKeyTriggered(ist::KEY_F1)) {
        m_active = !m_active;
    }

    if(isActive()) {
        const InputState &is = *atomicGetSystemInputs();
        if(is.isButtonTriggered(0)) {
            m_root->handleEvent(ist::IParamNode::Event_Action);
        }
        if(is.isButtonTriggered(1)) {
            m_root->handleEvent(ist::IParamNode::Event_Cancel);
        }
        if(is.isDirectionTriggered(InputState::Dir_Down)) {
            m_root->handleEvent(ist::IParamNode::Event_Down);
        }
        if(is.isDirectionTriggered(InputState::Dir_Up)) {
            m_root->handleEvent(ist::IParamNode::Event_Up);
        }
        if(is.isDirectionTriggered(InputState::Dir_Left)) {
            m_root->handleEvent(ist::IParamNode::Event_Backward);
        }
        if(is.isDirectionTriggered(InputState::Dir_Right)) {
            m_root->handleEvent(ist::IParamNode::Event_Forward);
        }
    }
}

void DebugMenu::draw()
{
    if(!isActive()) { return; }

    IFontRenderer *fr = atomicGetFontRenderer();
    ist::IParamNode *node = getRoot();
    vec2 base = vec2(10.0f, 100.0f);
    char buf_name[128];
    char buf_value[128];
    while(node) {
        ist::IParamNode *next_node = NULL;
        for(uint32 i=0; i<node->getNumChildren(); ++i) {
            ist::IParamNode *c = node->getChild(i);
            vec4 color = vec4(1.0f, 1.0f, 1.0f, 0.5f);
            if(c->isOpened()) {
                next_node = c;
                color = vec4(1.0f, 1.0f, 1.0f, 0.9f);
            }
            else if(i==node->getSelection()) {
                color = vec4(1.0f, 1.0f, 1.0f, 0.9f);
            }

            c->printName(buf_name, _countof(buf_name));
            c->printValue(buf_value, _countof(buf_value));
            vec2 s = fr->computeTextSize(buf_value);
            fr->setColor(color);
            fr->addText(base+vec2(0.0f, 20.0f*i), buf_name);
            fr->addText(base+vec2(200.0f-s.x, 20.0f*i), buf_value);
        }
        node = next_node;
        base += vec2(210.0f, 0.0f);
    }
}

ist::IParamNode* DebugMenu::getRoot()
{
    return m_root;
}

} // namespace atomic

#endif // atomic_enable_debug_menu
