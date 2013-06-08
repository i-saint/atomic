#ifndef atm_Game_AtomicApplication_h
#define atm_Game_AtomicApplication_h
#include "Input.h"

namespace atm {

class AtomicGame;
class SoundThread;
struct GameStartConfig;

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
    bool bg_multiresolution;
    bool light_multiresolution;
    bool show_text;
    bool show_bloodstain;
    bool output_replay;
    bool debug_show_grid;
    bool debug_show_distance;
    int32 debug_show_gbuffer;
    int32 debug_show_lights;
    int32 debug_show_resolution;
    bool sound_enable;
    float32 bgm_volume;
    float32 se_volume;
    int language;
    PlayerName name;

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
