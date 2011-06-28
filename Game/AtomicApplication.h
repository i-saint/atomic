#ifndef __atomic_AtomicApplication__
#define __atomic_AtomicApplication__
namespace atomic {

class AtomicGame;

class AtomicApplication : public ist::Application
{
typedef ist::Application super;
private:
    bool m_request_exit;
    AtomicGame *m_game;

public:
    static AtomicApplication* getInstance();

public:
    AtomicApplication();
    ~AtomicApplication();
    virtual bool Initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen=false);
    virtual void Finalize();
    virtual void mainLoop();
    virtual int handleWindowMessage(const ist::WindowMessage& wm);

    void update();
    void draw();
};


#define GetWindowWidth() AtomicApplication::getInstance()->getWindowWidth()
#define GetWindowHeight() AtomicApplication::getInstance()->getWindowHeight()

} // namespace atomic
#endif __atomic_AtomicApplication__
