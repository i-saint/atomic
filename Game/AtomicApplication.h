#ifndef __atomic_AtomicApplication__
#define __atomic_AtomicApplication__
namespace atomic {


class AtomicGame;
class AtomicRenderThread;

class AtomicApplication : public ist::Application
{
typedef ist::Application super;
private:
    bool m_request_exit;
    AtomicGame *m_game;
    AtomicRenderThread *m_draw_thread;

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

    // •`‰æƒXƒŒƒbƒh‚©‚çŒÄ‚Î‚ê‚é
    void drawCallback();

    AtomicGame* getGame() { return m_game; }
};


#define atomicGetApplication()          AtomicApplication::getInstance()
#define atomicGetGame()                 atomicGetApplication()->getGame()
#define atomicWaitForDrawComplete()     atomicGetApplication()->waitForDrawComplete()
#define atomicKickDraw()                atomicGetApplication()->kickDraw()

#define atomicGetWindowWidth()          atomicGetApplication()->getWindowWidth()
#define atomicGetWindowHeight()         atomicGetApplication()->getWindowHeight()
#define atomicGetWindowAspectRatio()    (float(atomicGetWindowWidth())/float(atomicGetWindowHeight()))

} // namespace atomic
#endif __atomic_AtomicApplication__
