#ifndef __atomic_Game_AtomicApplication__
#define __atomic_Game_AtomicApplication__
#include "Input.h"

namespace atomic {

class AtomicGame;
class AtomicRenderingThread;
class SoundThread;

struct AtomicConfig
{
    ivec2 window_pos;
    ivec2 window_size;
    float32 sound_volume; 
    bool fullscreen;
    bool posteffect_bloom;
    bool posteffect_antialias;

    AtomicConfig();
    bool readFromFile(const char* filepath);
    bool writeToFile(const char* filepath);
};


class AtomicApplication : public ist::Application
{
typedef ist::Application super;
private:
    AtomicGame              *m_game;
    AtomicRenderingThread   *m_renderng_thread;
    InputState        m_inputs;

    AtomicConfig            m_config;

    bool m_request_exit;

public:
    static AtomicApplication* getInstance();

public:
    AtomicApplication();
    ~AtomicApplication();
    virtual bool initialize(int argc, char *argv[]);
    virtual void finalize();

    virtual void mainLoop();
    virtual void updateInput();

    virtual int handleWindowMessage(const ist::WindowMessage& wm);
    void handleError(ERROR_CODE e);
    void handleCommandLine(const wchar_t* command, size_t command_len);

    void waitForDrawComplete();
    void kickDraw();

    // 描画スレッドから呼ばれる
    void drawCallback();

    AtomicGame* getGame()                           { return m_game; }
    const InputState* getSystemInputs() const { return &m_inputs; }
    AtomicConfig* getConfig()                       { return &m_config; }

    float32 getAverageFPS() const;
};


#define atomicGetApplication()          AtomicApplication::getInstance()
#define atomicGetGame()                 atomicGetApplication()->getGame()
#define atomicGetSystemInputs()         atomicGetApplication()->getSystemInputs()
#define atomicWaitForDrawComplete()     atomicGetApplication()->waitForDrawComplete()
#define atomicKickDraw()                atomicGetApplication()->kickDraw()

#define atomicGetConfig()               atomicGetApplication()->getConfig()
#define atomicGetWindowWidth()          atomicGetApplication()->getWindowWidth()
#define atomicGetWindowHeight()         atomicGetApplication()->getWindowHeight()
#define atomicGetWindowAspectRatio()    (float(atomicGetWindowWidth())/float(atomicGetWindowHeight()))

} // namespace atomic
#endif __atomic_Game_AtomicApplication__
