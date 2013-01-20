#ifndef atomic_Game_Debug_DebugMenu_h
#define atomic_Game_Debug_DebugMenu_h

#ifdef atomic_enable_debug_menu

namespace atomic {

class DebugMenu
{
public:
    static void initializeInstance();
    static void finalizeInstance();
    static DebugMenu* getInstance();

public:
    DebugMenu();
    ~DebugMenu();
    void update();
    void draw();

    bool isActive() const { return m_active; }
    ist::IParamNode* getRoot();

private:
    static DebugMenu *s_inst;

    ist::IParamNode *m_root;
    bool m_active;
};

} // namespace atomic

#define atomicDbgInitializeDebugMenu()                          DebugMenu::initializeInstance()
#define atomicDbgFinalizeDebugMenu()                            DebugMenu::finalizeInstance()
#define atomicDbgGetDebugMenu()                                 DebugMenu::getInstance()
#define atomicDbgDebugMenuUpdate()                              DebugMenu::getInstance()->update()
#define atomicDbgDebugMenuDraw()                                DebugMenu::getInstance()->draw()
#define atomicDbgDebugMenuIsActive()                            DebugMenu::getInstance()->isActive()
#define atomicDbgAddParamNode_F32(Name, Ptr, Min, Max, Step)    DebugMenu::getInstance()->getRoot()->addChildByPath(Name, istNew(ist::ParamNodeF32)("", Ptr, Min, Max, Step))
#define atomicDbgAddParamNode_I32(Name, Ptr, Min, Max, Step)    DebugMenu::getInstance()->getRoot()->addChildByPath(Name, istNew(ist::ParamNodeI32)("", Ptr, Min, Max, Step))
#define atomicDbgAddParamNode_U32(Name, Ptr, Min, Max, Step)    DebugMenu::getInstance()->getRoot()->addChildByPath(Name, istNew(ist::ParamNodeU32)("", Ptr, Min, Max, Step))
#define atomicDbgAddParamNode_Bool(Name, Ptr)                   DebugMenu::getInstance()->getRoot()->addChildByPath(Name, istNew(ist::ParamNodeBool)("", Ptr))
#define atomicDbgAddFunctionNode()                              

#else // atomic_enable_debug_menu

#define atomicDbgInitializeDebugMenu()                          
#define atomicDbgFinalizeDebugMenu()                            
#define atomicDbgGetDebugMenu()                                 
#define atomicDbgDebugMenuUpdate()                              
#define atomicDbgDebugMenuDraw()                                
#define atomicDbgDebugMenuIsActive()                            false
#define atomicDbgAddParamNode_F32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_I32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_U32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_Bool(Name, Ptr)                   
#define atomicDbgAddFunctionNode()                              

#endif // atomic_enable_debug_menu

#endif // atomic_Game_Debug_DebugMenu_h
