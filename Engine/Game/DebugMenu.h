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
#define atomicDbgDeleteParamNode(Name)                          DebugMenu::getInstance()->getRoot()->deleteChildByPath(Name);
#define atomicDbgAddParamNode(Name, Node, ...)\
    DebugMenu::getInstance()->getRoot()->addChildByPath(Name, istNew(ist::ParamNodeBase)(__VA_ARGS__))
#define atomicDbgAddParamNodeP(Name, Type, Ptr, ...)\
    DebugMenu::getInstance()->getRoot()->addChildByPath(Name, istNew(ist::TParamNode<Type>)(istNew(ist::TValueUpdaterP<Type>)(Ptr), __VA_ARGS__))
#define atomicDbgAddParamNodeM(Name, Type, Obj, Getter, Setter, ...)\
    DebugMenu::getInstance()->getRoot()->addChildByPath(Name, istNew(ist::TParamNode<Type>)(istNew(ist::TValueUpdaterM<Type>)(std::bind(Getter, Obj), std::bind(Setter, Obj, std::placeholders::_1)), __VA_ARGS__))
#define atomicDbgAddFunctionNode()                              

#else // atomic_enable_debug_menu

#define atomicDbgInitializeDebugMenu()                          
#define atomicDbgFinalizeDebugMenu()                            
#define atomicDbgGetDebugMenu()                                 
#define atomicDbgDebugMenuUpdate()                              
#define atomicDbgDebugMenuDraw()                                
#define atomicDbgDebugMenuIsActive()                            false
#define atomicDbgDeleteParamNode(Name)                          
#define atomicDbgAddParamNode(...)                              
#define atomicDbgAddParamNodeP(...)                             
#define atomicDbgAddParamNodeM(...)                             
#define atomicDbgAddFunctionNode()                              

#endif // atomic_enable_debug_menu

#endif // atomic_Game_Debug_DebugMenu_h
