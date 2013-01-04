#ifndef atomic_Game_DebugConsole_h
#define atomic_Game_DebugConsole_h
namespace atomic {

    class DebugConsole
    {
    private:
        stl::list<stl::wstring> m_history;
        stl::wstring m_command;

    public:
        void beforeDraw();
        void draw();
    };

} // namespace atomic
#endif // atomic_Game_DebugConsole_h
