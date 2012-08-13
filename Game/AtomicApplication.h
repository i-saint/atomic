#ifndef __atomic_Game_AtomicApplication__
#define __atomic_Game_AtomicApplication__
#include "Input.h"

namespace atomic {

class AtomicGame;
class SoundThread;

struct AtomicConfig
{
    ivec2 window_pos;
    ivec2 window_size;
    bool fullscreen;
    bool vsync;
    bool pause;
    bool posteffect_microscopic;
    bool posteffect_bloom;
    bool posteffect_antialias;
    bool multiresolution;
    bool show_text;
    bool show_bloodstain;
    bool debug_show_grid;
    bool debug_show_distance;
    int32 debug_show_gbuffer;
    int32 debug_show_lights;
    int32 debug_show_resolution;
    bool sound_enable;
    float32 bgm_volume;
    float32 se_volume;
    int language;

    AtomicConfig();
    bool readFromFile(const char* filepath);
    bool writeToFile(const char* filepath);
};


class AtomicApplication : public ist::Application
{
typedef ist::Application super;
private:
    AtomicGame              *m_game;
    InputState              m_inputs;

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
    void handleError(ATOMIC_ERROR e);
    int32 handleCommandLine(const stl::wstring &command);

    // 描画スレッドから呼ばれる
    void drawCallback();

    void requestExit()                          { m_request_exit=true; }
    AtomicGame* getGame()                       { return m_game; }
    const InputState* getSystemInputs() const   { return &m_inputs; }
    AtomicConfig* getConfig()                   { return &m_config; }
};


#define atomicGetApplication()          AtomicApplication::getInstance()
#define atomicGetGame()                 atomicGetApplication()->getGame()
#define atomicGetSystemInputs()         atomicGetApplication()->getSystemInputs()

#define atomicGetConfig()               atomicGetApplication()->getConfig()
#define atomicGetWindowSize()           atomicGetApplication()->getWindowSize()

} // namespace atomic
#endif __atomic_Game_AtomicApplication__
