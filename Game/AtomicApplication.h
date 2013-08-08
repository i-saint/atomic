#ifndef atm_Game_AtomicApplication_h
#define atm_Game_AtomicApplication_h
#include "Input.h"

namespace atm {

class AtomicGame;
class SoundThread;
struct GameStartConfig;

enum atmGraphicsLevel {
    atmE_Graphics_Low,
    atmE_Graphics_Medium,
    atmE_Graphics_High,
    atmE_Graphics_Custom,
};

enum atmLightingLevel {
    atmE_Lighting_Low,
    atmE_Lighting_Medium,
    atmE_Lighting_High,
};

enum atmBGResolution {
    atmE_BGResolution_x1,
    atmE_BGResolution_x2,
    atmE_BGResolution_x4,
    atmE_BGResolution_x8,
    atmE_BGNone,
};

struct atmAPI AtomicConfig
{
    ivec2 window_pos;
    ivec2 window_size;
    bool fullscreen;
    bool vsync;
    bool unlimit_gamespeed;
    bool pause;
    bool posteffect_microscopic;
    bool posteffect_bloom;
    bool posteffect_antialias;
    bool bg_multiresolution;
    bool light_multiresolution;
    bool show_text;
    bool show_bloodstain;
    bool output_replay;
    bool sound_enable;
    float32 bgm_volume;
    float32 se_volume;
    int32 language;
    int32 bg_level;
    int32 lighting_level;
    int32 graphics_level;
    uint32 leveleditor_port;
    PlayerName name;

    bool debug_show_grid;
    bool debug_show_distance;
    int32 debug_show_gbuffer;
    int32 debug_show_lights;
    int32 debug_show_resolution;

    AtomicConfig();
    void setupDebugMenu();
    bool readFromFile(const char* filepath);
    bool writeToFile(const char* filepath);
};


class atmAPI AtomicApplication : public ist::Application
{
typedef ist::Application super;
public:
    static AtomicApplication* getInstance();

    AtomicApplication();
    ~AtomicApplication();
    virtual bool initialize(int argc, char *argv[]);
    virtual void finalize();

    virtual void mainLoop();
    virtual void updateInput();
    void update();
    void draw();

    bool handleWindowMessage(const ist::WM_Base& wm);
    void handleError(ErrorCode e);

    // 描画スレッドから呼ばれる
    void drawCallback();

    void requestStartGame(const GameStartConfig &conf);
    void requestReturnToTitleScreen();
    void requestExit();
    AtomicGame* getGame();
    const InputState* getSystemInputs() const;
    AtomicConfig* getConfig();

    const ist::KeyboardState&   getKeyboardState() const;
    const ist::MouseState&      getMouseState() const;
    const ist::ControllerState& getControllerState() const;

#ifdef atm_enable_DebugLog
    void printDebugLog(const char *format, ...);
#endif // atm_enable_DebugLog

private:
    void registerCommands();

    typedef ist::Application::WMHandler WMHandler;
    WMHandler                   m_wnhandler;
    tbb::task_scheduler_init    m_tbb_init;

    ist::IKeyboardDevice   *m_keyboard;
    ist::IMouseDevice      *m_mouse;
    ist::IControlerDevice  *m_controller;

    AtomicGame     *m_game;
    InputState      m_inputs;
    AtomicConfig    m_config;
    bool            m_request_exit;
    bool            m_request_title;
#ifdef atm_enable_DebugLog
    FILE           *m_log;
#endif // atm_enable_DebugLog
    ist::vector<HMODULE> m_dlls;
};

void atmPause(bool v);
bool atmIsPaused();
void atmPauseAndShowPauseMenu();

} // namespace atm

#define atmGetApplication()          AtomicApplication::getInstance()
#define atmGetGame()                 atmGetApplication()->getGame()
#define atmGetSystemInputs()         atmGetApplication()->getSystemInputs()
#define atmGetConfig()               atmGetApplication()->getConfig()
#define atmGetWindowSize()           atmGetApplication()->getWindowSize()
#ifdef atm_enable_DebugLog
#   define atmDebugLog(...)          atmGetApplication()->printDebugLog(__VA_ARGS__)
#else  // atm_enable_DebugLog
#   define atmDebugLog(...)          
#endif // atm_enable_DebugLog

#define atmRequestReturnToTitleScreen() atmGetApplication()->requestReturnToTitleScreen()
#define atmRequestExit()                atmGetApplication()->requestExit()

#endif atm_Game_AtomicApplication_h
