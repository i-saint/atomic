#ifndef atomic_Game_Debug_DebugMenu_h
#define atomic_Game_Debug_DebugMenu_h

namespace atomic {

class DebugMenu
{
public:
    DebugMenu();
    ~DebugMenu();
    void update();
    void draw();

    ist::IParamNode* getRoot();

private:
    ist::IParamNode *m_root;
};

} // namespace atomic


#define atomicDebugAddBoolNode(Name, Ptr)
#define atomicDebugAddValueNode(Name, Ptr, Min, Max, Step)
#define atomicDebugAddFunctionNode()

#endif // atomic_Game_Debug_DebugMenu_h
