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

    ist::IParamNode* getRoot();

private:
    static DebugMenu *s_inst;
    ist::IParamNode *m_root;
};

} // namespace atomic

#define atomicDbgInitializeDebugMenu()                          DebugMenu::initializeInstance()
#define atomicDbgFinalizeDebugMenu()                            DebugMenu::finalizeInstance()
#define atomicDbgGetDebugMenu()                                 DebugMenu::getInstance()
#define atomicDbgAddParamNode_F32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_I32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_U32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_Bool(Name, Ptr)                   
#define atomicDbgAddFunctionNode()                              

#else // atomic_enable_debug_menu

#define atomicDbgInitializeDebugMenu()                          
#define atomicDbgFinalizeDebugMenu()                            
#define atomicDbgGetDebugMenu()                                 
#define atomicDbgAddParamNode_F32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_I32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_U32(Name, Ptr, Min, Max, Step)    
#define atomicDbgAddParamNode_Bool(Name, Ptr)                   
#define atomicDbgAddFunctionNode()                              

#endif // atomic_enable_debug_menu

#endif // atomic_Game_Debug_DebugMenu_h
