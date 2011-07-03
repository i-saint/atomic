#ifndef __atomic_AtomicApplication__
#define __atomic_AtomicApplication__
namespace atomic {


class AtomicGame;
class AtomicDrawThread;

class AtomicApplication : public ist::Application
{
typedef ist::Application super;
private:
    bool m_request_exit;
    AtomicGame *m_game;
    AtomicDrawThread *m_draw_thread;

public:
    static AtomicApplication* getInstance();

public:
    AtomicApplication();
    ~AtomicApplication();
    virtual bool initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen=false);
    virtual void finalize();
    virtual void mainLoop();
    virtual int handleWindowMessage(const ist::WindowMessage& wm);

    void waitForDrawComplete();
    void kickDraw();
    AtomicGame* getGame() { return m_game; }
};


#define WaitForDrawComplete() AtomicApplication::getInstance()->waitForDrawComplete()
#define KickDraw() AtomicApplication::getInstance()->kickDraw()

#define GetWindowWidth() AtomicApplication::getInstance()->getWindowWidth()
#define GetWindowHeight() AtomicApplication::getInstance()->getWindowHeight()
#define GetWindowAspectRatio() (float(GetWindowWidth())/float(GetWindowHeight()))

} // namespace atomic
#endif __atomic_AtomicApplication__
