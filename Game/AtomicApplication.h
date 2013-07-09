#ifndef atm_Game_AtomicApplication_h
#define atm_Game_AtomicApplication_h
#include "Input.h"

namespace atm {

class AtomicGame;
class SoundThread;
struct GameStartConfig;

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

struct istInterModule AtomicConfig
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
    int32 bg_level;
    bool bg_multiresolution;
    bool light_multiresolution;
    bool show_text;
    bool show_bloodstain;
    bool output_replay;
    bool sound_enable;
    float32 bgm_volume;
    float32 se_volume;
    int32 language;
    int32 lighting;
    uint32 leveleditor_port;
    PlayerName name;

    bool editmode;

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


class istInterModule AtomicApplication : public ist::Application
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
    void handleError(ATOMIC_ERROR e);

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

#ifdef atm_enable_debug_log
    void printDebugLog(const char *format, ...);
#endif // atm_enable_debug_log

private:
    void registerCommands();

    istMemberPtrDecl(Members) m;
};

} // namespace atm


#define atmGetApplication()          AtomicApplication::getInstance()
#define atmGetGame()                 atmGetApplication()->getGame()
#define atmGetSystemInputs()         atmGetApplication()->getSystemInputs()
#define atmGetConfig()               atmGetApplication()->getConfig()
#define atmGetWindowSize()           atmGetApplication()->getWindowSize()
#ifdef atm_enable_debug_log
#   define atmDebugLog(...)          atmGetApplication()->printDebugLog(__VA_ARGS__)
#else  // atm_enable_debug_log
#   define atmDebugLog(...)          
#endif // atm_enable_debug_log

#endif atm_Game_AtomicApplication_h
