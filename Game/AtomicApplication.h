#ifndef __atomic_AtomicApplication_h__
#define __atomic_AtomicApplication_h__
namespace atomic {


class AtomicGame;
class AtomicRenderingThread;
class AtomicSoundThread;

struct AtomicConfig
{
    ivec2 window_pos;
    ivec2 window_size;
    float32 sound_volume; 
    bool fullscreen;
    bool posteffect_bloom;
    bool posteffect_motionblur;
    bool posteffect_antialias;

    AtomicConfig();
    bool readFromFile(const char* filepath);
    bool writeToFile(const char* filepath);
};

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
    AtomicGame              *m_game;
    AtomicRenderingThread   *m_renderng_thread;
    AtomicSoundThread       *m_sound_thread;

    AtomicConfig            m_config;
    AtomicInputState        m_inputs;
    bool m_request_exit;

public:
    static AtomicApplication* getInstance();

public:
    AtomicApplication();
    ~AtomicApplication();
    virtual bool initialize();
    virtual void finalize();

    virtual void mainLoop();
    virtual void updateInput();
    virtual int handleWindowMessage(const ist::WindowMessage& wm);
    void handleError(ERROR_CODE e);
    void handleCommandLine(const wchar_t* command, size_t command_len);

    void waitForDrawComplete();
    void kickDraw();

    // •`‰æƒXƒŒƒbƒh‚©‚çŒÄ‚Î‚ê‚é
    void drawCallback();

    AtomicGame* getGame() { return m_game; }
    AtomicInputState* getInputs() { return &m_inputs; }
    AtomicConfig* getConfig() { return &m_config; }

    float32 getAverageFPS() const;
};


#define atomicGetApplication()          AtomicApplication::getInstance()
#define atomicGetGame()                 atomicGetApplication()->getGame()
#define atomicGetInputs()               atomicGetApplication()->getInputs()
#define atomicWaitForDrawComplete()     atomicGetApplication()->waitForDrawComplete()
#define atomicKickDraw()                atomicGetApplication()->kickDraw()

#define atomicGetConfig()               atomicGetApplication()->getConfig()
#define atomicGetWindowWidth()          atomicGetApplication()->getWindowWidth()
#define atomicGetWindowHeight()         atomicGetApplication()->getWindowHeight()
#define atomicGetWindowAspectRatio()    (float(atomicGetWindowWidth())/float(atomicGetWindowHeight()))

} // namespace atomic
#endif __atomic_AtomicApplication_h__
