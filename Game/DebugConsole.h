#ifndef __atomic_Game_DebugConsole__
#define __atomic_Game_DebugConsole__
namespace atomic {

    class DebugConsole
    {
    private:
        stl::vector<stl::wstring> m_history;
        stl::wstring m_command;

    public:
        void beforeDraw();
        void draw();
    };

} // namespace atomic
#endif // __atomic_Game_DebugConsole__
