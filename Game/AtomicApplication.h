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


#define atomicWaitForDrawComplete()     AtomicApplication::getInstance()->waitForDrawComplete()
#define atomicKickDraw()                AtomicApplication::getInstance()->kickDraw()

#define atomicGetWindowWidth()          AtomicApplication::getInstance()->getWindowWidth()
#define atomicGetWindowHeight()         AtomicApplication::getInstance()->getWindowHeight()
#define atomicGetWindowAspectRatio()    (float(atomicGetWindowWidth())/float(atomicGetWindowHeight()))

} // namespace atomic
#endif __atomic_AtomicApplication__
