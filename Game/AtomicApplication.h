#ifndef __atomic_AtomicApplication_h__
#define __atomic_AtomicApplication_h__
namespace atomic {


class AtomicGame;
class AtomicRenderingThread;
class AtomicSoundThread;

struct AtomicInputState
{
private:
    float2 m_move;
    int32 m_buttons;

public:
    AtomicInputState()
    {
        m_move = make_float2(0.0f);
        m_buttons = 0;
    }

    float2 getMove() const { return m_move; }
    int32 getButtons() const { return m_buttons; }

    void setMove(float2 v) { m_move=v; }
    void setButtons(int32 v) { m_buttons=v; }
};

class AtomicApplication : public ist::Application
{
typedef ist::Application super;
private:
    AtomicGame *m_game;
    AtomicRenderingThread *m_renderng_thread;
    AtomicSoundThread *m_sound_thread;

    AtomicInputState m_inputs;
    bool m_request_exit;

public:
    static AtomicApplication* getInstance();

public:
    AtomicApplication();
    ~AtomicApplication();
    virtual bool initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen=false);
    virtual void finalize();

    virtual void mainLoop();
    virtual void updateInput();
    virtual int handleWindowMessage(const ist::WindowMessage& wm);

    void waitForDrawComplete();
    void kickDraw();

    // •`‰æƒXƒŒƒbƒh‚©‚çŒÄ‚Î‚ê‚é
    void drawCallback();

    AtomicGame* getGame() { return m_game; }
    AtomicInputState* getInputs() { return &m_inputs; }
};


#define atomicGetApplication()          AtomicApplication::getInstance()
#define atomicGetGame()                 atomicGetApplication()->getGame()
#define atomicGetInputs()               atomicGetApplication()->getInputs()
#define atomicWaitForDrawComplete()     atomicGetApplication()->waitForDrawComplete()
#define atomicKickDraw()                atomicGetApplication()->kickDraw()

#define atomicGetWindowWidth()          atomicGetApplication()->getWindowWidth()
#define atomicGetWindowHeight()         atomicGetApplication()->getWindowHeight()
#define atomicGetWindowAspectRatio()    (float(atomicGetWindowWidth())/float(atomicGetWindowHeight()))

} // namespace atomic
#endif __atomic_AtomicApplication_h__
