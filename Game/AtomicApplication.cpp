#include "stdafx.h"
#include "ist/iui.h"
#include "types.h"
#include "AtomicGame.h"
#include "AtomicApplication.h"
#include "Text.h"
#include "Graphics/AtomicRenderingSystem.h"
#include "Game/World.h"
#include "Sound/AtomicSound.h"
#include "Graphics/Renderer.h"
#include "Network/WebServer.h"
#include "Network/GameServer.h"
#include "Network/GameClient.h"
#include "Util.h"

#define ATOMIC_CONFIG_FILE_PATH "atomic.conf"

namespace atm {

void atmInitializeCrashReporter();
void atmFinalizeCrashReporter();
iui::RootWindow*    atmCreateRootWindow();
class TitleWindow;
iui::Widget*        atmGetTitleWindow();


AtomicConfig::AtomicConfig()
{
    window_pos              = ivec2(0, 0);
    window_size             = ivec2(1280, 768);
    fullscreen              = false;
    vsync                   = false;
    unlimit_gamespeed       = false;
    pause                   = false;
    posteffect_microscopic  = false;
    posteffect_bloom        = true;
    posteffect_antialias    = false;
    bg_level           = atmE_BGResolution_x1;
    bg_multiresolution      = false;
    light_multiresolution   = false;
    show_text               = true;
    show_bloodstain         = true;
    output_replay           = true;
    sound_enable            = true;
    bgm_volume              = 0.3f;
    se_volume               = 0.3f;
    language                = lang_JP;
    lighting                = atmE_Lighting_Medium;
    leveleditor_port        = atm_Leveleditor_DefaultPort;
    wcscpy(name, L"atom");

    debug_show_grid         = false;
    debug_show_distance     = false;
    debug_show_gbuffer      = 0;
    debug_show_lights       = -1;
    debug_show_resolution   = 0;
}

bool AtomicConfig::readFromFile( const char* filepath )
{
    FILE *f = fopen(filepath, "r");
    if(!f) { return false; }

    char buf[256];
    uvec2 utmp;
    ivec2 itmp;
    vec2 ftmp;
    while(fgets(buf, 256, f)) {
        if(sscanf(buf, "window_pos = %d, %d", &itmp.x, &itmp.y)==2) { window_pos.x=itmp.x; window_pos.y=itmp.y; }
        if(sscanf(buf, "window_size = %d, %d", &itmp.x, &itmp.y)==2){ window_size.x=itmp.x; window_size.y=itmp.y; }
        if(sscanf(buf, "fullscreen = %d", &itmp.x)==1)              { fullscreen=itmp.x!=0; }
        if(sscanf(buf, "vsync = %d", &itmp.x)==1)                   { vsync=itmp.x!=0; }
        if(sscanf(buf, "unlimit_gamespeed = %d", &itmp.x)==1)       { unlimit_gamespeed=itmp.x!=0; }
        if(sscanf(buf, "posteffect_bloom = %d", &itmp.x)==1)        { posteffect_bloom=(itmp.x!=0); }
        if(sscanf(buf, "posteffect_antialias = %d", &itmp.x)==1)    { posteffect_antialias=(itmp.x!=0); }
        if(sscanf(buf, "bg_level = %d", &itmp.x)==1)                { bg_level=itmp.x; }
        if(sscanf(buf, "bg_multiresolution = %d", &itmp.x)==1)      { bg_multiresolution=(itmp.x!=0); }
        if(sscanf(buf, "light_multiresolution = %d", &itmp.x)==1)   { light_multiresolution=(itmp.x!=0); }
        if(sscanf(buf, "show_text = %d", &itmp.x)==1)               { show_text=(itmp.x!=0); }
        if(sscanf(buf, "show_bloodstain = %d", &itmp.x)==1)         { show_bloodstain=(itmp.x!=0); }
        if(sscanf(buf, "output_replay = %d", &itmp.x)==1)           { output_replay=(itmp.x!=0); }
        if(sscanf(buf, "sound_enable = %f", &itmp.x)==1)            { sound_enable=(itmp.x!=0); }
        if(sscanf(buf, "bgm_volume = %f", &ftmp.x)==1)              { bgm_volume=ftmp.x; }
        if(sscanf(buf, "se_volume = %f", &ftmp.x)==1)               { se_volume=ftmp.x; }
        if(sscanf(buf, "lighting = %d", &itmp.x)==1)                { lighting=itmp.x; }
        if(sscanf(buf, "leveleditor_port = %d", &utmp.x)==1)        { leveleditor_port=utmp.x; }
        if(sscanf(buf, "debug_show_grid = %d", &itmp.x)==1)         { debug_show_grid=(itmp.x!=0); }
        if(sscanf(buf, "debug_show_distance = %d", &itmp.x)==1)     { debug_show_distance=(itmp.x!=0); }
        if(sscanf(buf, "debug_show_resolution = %d", &itmp.x)==1)   { debug_show_resolution=(itmp.x!=0); }
    }
    fclose(f);
    return true;
}

bool AtomicConfig::writeToFile( const char* filepath )
{
    FILE *f = fopen(filepath, "w");
    if(!f) { return false; }

    fprintf(f, "window_pos = %d, %d\n",         window_pos.x, window_pos.y);
    fprintf(f, "window_size = %d, %d\n",        window_size.x, window_size.y);
    fprintf(f, "fullscreen = %d\n",             fullscreen);
    fprintf(f, "vsync = %d\n",                  vsync);
    fprintf(f, "unlimit_gamespeed = %d\n",      unlimit_gamespeed);
    fprintf(f, "posteffect_bloom = %d\n",       posteffect_bloom);
    fprintf(f, "posteffect_antialias = %d\n",   posteffect_antialias);
    fprintf(f, "bg_level = %d\n",               bg_level);
    fprintf(f, "bg_multiresolution = %d\n",     bg_multiresolution);
    fprintf(f, "light_multiresolution = %d\n",  light_multiresolution);
    fprintf(f, "show_text = %d\n",              show_text);
    fprintf(f, "show_bloodstain = %d\n",        show_bloodstain);
    fprintf(f, "output_replay = %d\n",          output_replay);
    fprintf(f, "sound_enable = %d\n",           sound_enable);
    fprintf(f, "bgm_volume = %f\n",             bgm_volume);
    fprintf(f, "se_volume = %f\n",              se_volume);
    fprintf(f, "lighting = %d\n",               lighting);
    fprintf(f, "leveleditor_port = %d",         leveleditor_port);
    fprintf(f, "debug_show_grid = %d\n",        debug_show_grid);
    fprintf(f, "debug_show_distance = %d\n",    debug_show_distance);
    fprintf(f, "debug_show_resolution = %d\n",  debug_show_resolution);
    fclose(f);
    return true;
}

void AtomicConfig::setupDebugMenu()
{
    wdmAddNode("Config/VSync",  &vsync);
    wdmAddNode("Config/ShowText",   &show_text);
    wdmAddNode("Config/UnlimitGameSpeed",   &unlimit_gamespeed);
    wdmAddNode("Config/PostEffect_Bloom",     &posteffect_bloom);
    wdmAddNode("Config/PostEffect_Antialias", &posteffect_antialias);
    wdmAddNode("Config/Lighting", &lighting, (int32)atmE_Lighting_Low, (int32)atmE_Lighting_High);
    wdmAddNode("Config/BG_Level", &bg_level, (int32)atmE_BGResolution_x1, (int32)atmE_BGNone);
    wdmAddNode("Config/BG_Multiresolution", &bg_multiresolution);
}



void AtomicApplication::requestExit()                       { m_request_exit=true; }
void AtomicApplication::requestReturnToTitleScreen()        { m_request_title=true; }
AtomicGame* AtomicApplication::getGame()                    { return m_game; }
const InputState* AtomicApplication::getSystemInputs() const{ return &m_inputs; }
AtomicConfig* AtomicApplication::getConfig()                { return &m_config; }



AtomicApplication* AtomicApplication::getInstance() { return static_cast<AtomicApplication*>(ist::Application::getInstance()); }

AtomicApplication::AtomicApplication()
    : m_keyboard(nullptr)
    , m_mouse(nullptr)
    , m_controller(nullptr)
    , m_game(nullptr)
    , m_request_exit(false)
    , m_request_title(false)
{
    m_wnhandler = std::bind(&AtomicApplication::handleWindowMessage, this, std::placeholders::_1);
    addMessageHandler(&m_wnhandler);

#ifdef atm_enable_debug_log
    m_log = fopen("atomic.log", "wb");
#endif // atm_enable_debug_log
}

AtomicApplication::~AtomicApplication()
{
#ifdef atm_enable_debug_log
    if(m_log!=nullptr) {
        fclose(m_log);
    }
#endif // atm_enable_debug_log

    eraseMessageHandler(&m_wnhandler);
}

bool AtomicApplication::initialize(int argc, char *argv[])
{
    AtomicConfig &conf = m_config;
    conf.readFromFile(ATOMIC_CONFIG_FILE_PATH);
    if(conf.window_pos.x >= 30000) { conf.window_pos.x = 0; }
    if(conf.window_pos.y >= 30000) { conf.window_pos.y = 0; }
    if(conf.window_size.x < 320 || conf.window_size.x < 240) { conf.window_size = ivec2(1024, 768); }

    m_keyboard     = ist::CreateKeyboardDevice();
    m_mouse        = ist::CreateMouseDevice();
    m_controller   = ist::CreateControllerDevice();

#ifdef atm_enable_shader_live_edit
    ::AllocConsole();
#endif // atm_enable_shader_live_edit
    wdmInitialize();
    istTaskSchedulerInitialize();

    // initialize debug menu
    conf.setupDebugMenu();

    // console
    istCommandlineInitialize();
    istCommandlineConsoleInitialize();

    // create window
    ivec2 wpos = conf.window_pos;
    ivec2 wsize = conf.window_size;
    if(!super::initialize(wpos, wsize, L"atomic", conf.fullscreen))
    {
        return false;
    }
    // ui
    iuiInitialize();
    iuiGetSystem()->setRootWindow(atmCreateRootWindow());
    iuiGetSystem()->setScreen(768.0f*istGetAspectRatio(), 768.0f);

    // start rendering thread
    AtomicRenderingSystem::initializeInstance();

    // initialize sound
    AtomicSound::initializeInstance();

    //// create game
    //GameStartConfig gconf;
    //if(argc > 1) {
    //    gconf.gmode = GameStartConfig::GM_Replay;
    //    gconf.path_to_replay = argv[1];
    //}
    //m_game = istNew(AtomicGame)(gconf);

    // start server
    Poco::ThreadPool::defaultPool().addCapacity(8);
    atmGameServerInitialize();
    atmGameClientInitialize();
    atmWebServerInitialize();

    atmGameClientConnect("localhost", atm_GameServer_DefaultPort);

    registerCommands();

    return true;
}

void AtomicApplication::finalize()
{
    m_config.writeToFile(ATOMIC_CONFIG_FILE_PATH);

    atmWebServerFinalize();
    atmGameClientFinalize();
    atmGameServerFinalize();
    Poco::ThreadPool::defaultPool().joinAll();

    istSafeDelete(m_game);

    AtomicSound::finalizeInstance();
    AtomicRenderingSystem::finalizeInstance();
    iuiFinalize();
    super::finalize();

    istCommandlineConsoleFinalize();
    istCommandlineFinalize();

    istTaskSchedulerFinalize();

    istSafeRelease(m_controller);
    istSafeRelease(m_mouse);
    istSafeRelease(m_keyboard);

    FinalizeText();
    atmFinalizeCrashReporter();
    istPoolRelease();
    wdmFinalize();
}

void AtomicApplication::mainLoop()
{

    ist::Timer pc;
    const float32 delay = 16.666f;
    const float32 dt = 1.0f;

    while(!m_request_exit)
    {
        dpUpdate();
        wdmFlush();
        istCommandlineFlush();
        translateMessage();
        update();

        AtomicGame *game = m_game;
        if(game) {
            game->frameBegin();
            game->update(dt);
            game->asyncupdateBegin(dt);
            updateInput();
            draw();
            game->asyncupdateEnd();
            game->frameEnd();

            if(m_request_title) {
                m_request_title = false;
                istSafeDelete(m_game);
                ((iui::Widget*)atmGetTitleWindow())->setVisibility(true);
            }
        }
        else {
            updateInput();
            draw();
        }

        bool needs_sync = true;
        if( atmGetConfig()->unlimit_gamespeed || atmGetConfig()->vsync ||
            (game && (game->isUpdateSkipped() || game->isDrawSkipped()) ) ) {
            needs_sync = false;
        }
        if(needs_sync) {
            float32 remain = delay-pc.getElapsedMillisec();
            if(remain>0.0f) {
                ist::MicroSleep((uint32)std::max<float32>(remain*1000.0f, 0.0f));
            }
            pc.reset();
        }
    }
}

void AtomicApplication::update()
{
    istPoolUpdate();
    iuiUpdate();

    AtomicConfig &conf = m_config;
    if(getKeyboardState().isKeyTriggered(ist::KEY_F1)) {
        wdmOpenBrowser();
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F2)) {
        char url[256];
        sprintf(url, "http://localhost:%d", atmGetConfig()->leveleditor_port);
        ::ShellExecuteA(nullptr, "open", url, "", "", SW_SHOWDEFAULT);
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F3)) {
        conf.debug_show_gbuffer--;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F4)) {
        conf.debug_show_gbuffer++;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F5)) {
        conf.debug_show_lights--;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F6)) {
        conf.debug_show_lights++;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F7)) {
        conf.pause = !conf.pause;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F8)) {
        conf.bg_multiresolution = !conf.bg_multiresolution;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F9)) {
        conf.debug_show_resolution = !conf.debug_show_resolution;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_ESCAPE)) {
        if(m_game) {
            atmPauseAndShowPauseMenu();
        }
        else {
            m_request_exit = true;
        }
    }
    if(getKeyboardState().isKeyTriggered('7')) {
        atmGetRenderStates()->ShowMultiresolution = !atmGetRenderStates()->ShowMultiresolution;
    }
    if(getKeyboardState().isKeyPressed('8')) {
        float &p = atmGetLightPass()->getMultiresolutionParams().Threshold.x;
        p = clamp(p-0.001f, 0.0f, 1.0f);
    }
    if(getKeyboardState().isKeyPressed('9')) {
        float &p = atmGetLightPass()->getMultiresolutionParams().Threshold.x;
        p = clamp(p+0.001f, 0.0f, 1.0f);
    }
}

void AtomicApplication::draw()
{
    AtomicGame *game = m_game;
    if(game) {
        game->draw();
    }
    if(!game || !game->isDrawSkipped()) {
        atmKickDraw();
        atmWaitUntilDrawCallbackComplete();
    }
}

void AtomicApplication::requestStartGame(const GameStartConfig &conf)
{
    istSafeDelete(m_game);
    m_game = istNew(AtomicGame)();
    m_game->config(conf);
}


void AtomicApplication::updateInput()
{
    m_keyboard->update();
    m_mouse->update();
    m_controller->update();

    AtomicConfig &conf = m_config;

    RepMove move;
    RepButton buttons = 0;
    if(getControllerState().isButtonPressed(ist::ControllerState::Button_1)) { buttons |= 1<<0; }
    if(getControllerState().isButtonPressed(ist::ControllerState::Button_2)) { buttons |= 1<<1; }
    if(getControllerState().isButtonPressed(ist::ControllerState::Button_3)) { buttons |= 1<<2; }
    if(getControllerState().isButtonPressed(ist::ControllerState::Button_4)) { buttons |= 1<<3; }

    const ist::MouseState &mouse = getMouseState();
    if(mouse.isButtonPressed(ist::MouseState::Button_Left  )) { buttons |= 1<<0; }
    if(mouse.isButtonPressed(ist::MouseState::Button_Right )) { buttons |= 1<<1; }
    if(mouse.isButtonPressed(ist::MouseState::Button_Middle)) { buttons |= 1<<2; }

    const ist::KeyboardState &kb = getKeyboardState();
    if(kb.isKeyPressed('Z')) { buttons |= 1<<0; }
    if(kb.isKeyPressed('X')) { buttons |= 1<<1; }
    if(kb.isKeyPressed('C')) { buttons |= 1<<2; }
    if(kb.isKeyPressed('V')) { buttons |= 1<<3; }
    if(kb.isKeyPressed(ist::KEY_RIGHT)  || kb.isKeyPressed('D')) { move.x = INT16_MAX; }
    if(kb.isKeyPressed(ist::KEY_LEFT)   || kb.isKeyPressed('A')) { move.x =-INT16_MAX; }
    if(kb.isKeyPressed(ist::KEY_UP)     || kb.isKeyPressed('W')) { move.y = INT16_MAX; }
    if(kb.isKeyPressed(ist::KEY_DOWN)   || kb.isKeyPressed('S')) { move.y =-INT16_MAX; }

    {
        vec2 pos = getControllerState().getStick1();
        RepMove jpos(int16(pos.x*INT16_MAX), int16(pos.y*INT16_MAX));
        if(glm::length(jpos.toF())>0.4f) { move=jpos; }
    }
    m_inputs.update(RepInput(move, buttons));
}

bool AtomicApplication::handleWindowMessage(const ist::WM_Base& wm)
{
    switch(wm.type)
    {
    case ist::WMT_WindowClose:
        {
            m_request_exit = true;
        }
        return true;

    case ist::WMT_KeyUp:
        return true;

    case ist::WMT_WindowSize:
        {
            auto &mes = static_cast<const ist::WM_Window&>(wm);
            m_config.window_size = mes.window_size;
        }
        return true;

    case ist::WMT_WindowMove:
        {
            auto &mes = static_cast<const ist::WM_Window&>(wm);
            m_config.window_pos = mes.window_pos;
        }
        return true;


    case ist::WMT_IMEBegin:
        {
            istPrint(L"WMT_IMEBegin\n");
        }
        break;
    case ist::WMT_IMEEnd:
        {
            istPrint(L"WMT_IMEEnd\n");
        }
        break;
    case ist::WMT_IMEChar:
    case ist::WMT_IMEResult:
        {
            auto &mes = static_cast<const ist::WM_IME&>(wm);
            stl::wstring str(mes.text, mes.text_len);
            switch(wm.type) {
            case ist::WMT_IMEChar:   istPrint(L"WMT_IMEChar %s\n", str.c_str()); break;
            case ist::WMT_IMEResult: istPrint(L"WMT_IMEResult %s\n", str.c_str()); break;
            }
        }
        break;
    }

    return false;
}

void AtomicApplication::handleError(ErrorCode e)
{
    stl::wstring mes;
    switch(e) {
    case atmE_OpenGL_330NotSupported:   mes=GetText(txt_OpenGL_330NotSupported); break;
    }
    istShowMessageDialog(mes.c_str(), L"error", DLG_OK);
}


void AtomicApplication::drawCallback()
{
    AtomicRenderer::getInstance()->beforeDraw();
    if(m_game) {
        m_game->drawCallback();
    }
}



#ifdef atm_enable_debug_log
void AtomicApplication::printDebugLog( const char *format, ... )
{
    if(m_log==nullptr) { return; }
    va_list vl;
    va_start(vl, format);
    fprintf(m_log, "%d ", (uint32)atmGetFrame());
    vfprintf(m_log, format, vl);
    va_end(vl);
}
#endif // atm_enable_debug_log


void AtomicApplication::registerCommands()
{
    istCommandlineRegister("printPoolStates", &ist::PoolManager::printPoolStates);
}

const ist::KeyboardState& AtomicApplication::getKeyboardState() const
{
    return m_keyboard->getState();
}

const ist::MouseState& AtomicApplication::getMouseState() const
{
    return m_mouse->getState();
}

const ist::ControllerState& AtomicApplication::getControllerState() const
{
    return m_controller->getState();
}


void atmPause(bool v)   { atmGetConfig()->pause=v; }
bool atmIsPaused()      { return atmGetConfig()->pause; }

} // namespace atm
