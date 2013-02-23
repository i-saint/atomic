#include "stdafx.h"
#include "types.h"
#include "AtomicGame.h"
#include "AtomicApplication.h"
#include "Text.h"
#include "Graphics/AtomicRenderingSystem.h"
#include "Game/World.h"
#include "Game/DebugMenu.h"
#include "Sound/AtomicSound.h"
#include "Graphics/Renderer.h"
#include "Network/LevelEditorServer.h"
#include "Network/GameServer.h"
#include "Network/GameClient.h"
#include "Util.h"

#define ATOMIC_CONFIG_FILE_PATH "atomic.conf"

namespace atomic {

void InitializeCrashReporter();
void FinalizeCrashReporter();


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
    bg_multiresolution      = false;
    light_multiresolution   = false;
    show_text               = true;
    show_bloodstain         = true;
    output_replay           = true;
    debug_show_grid         = false;
    debug_show_distance     = false;
    debug_show_gbuffer      = 0;
    debug_show_lights       = -1;
    debug_show_resolution   = 0;
    sound_enable            = true;
    bgm_volume              = 0.3f;
    se_volume               = 0.3f;
    language                = LANG_JP;
    wcscpy(name, L"atom");
}

bool AtomicConfig::readFromFile( const char* filepath )
{
    FILE *f = fopen(filepath, "r");
    if(!f) { return false; }

    char buf[256];
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
        if(sscanf(buf, "bg_multiresolution = %d", &itmp.x)==1)      { bg_multiresolution=(itmp.x!=0); }
        if(sscanf(buf, "light_multiresolution = %d", &itmp.x)==1)   { light_multiresolution=(itmp.x!=0); }
        if(sscanf(buf, "show_text = %d", &itmp.x)==1)               { show_text=(itmp.x!=0); }
        if(sscanf(buf, "show_bloodstain = %d", &itmp.x)==1)         { show_bloodstain=(itmp.x!=0); }
        if(sscanf(buf, "output_replay = %d", &itmp.x)==1)           { output_replay=(itmp.x!=0); }
        if(sscanf(buf, "debug_show_grid = %d", &itmp.x)==1)         { debug_show_grid=(itmp.x!=0); }
        if(sscanf(buf, "debug_show_distance = %d", &itmp.x)==1)     { debug_show_distance=(itmp.x!=0); }
        if(sscanf(buf, "debug_show_resolution = %d", &itmp.x)==1)   { debug_show_resolution=(itmp.x!=0); }
        if(sscanf(buf, "sound_enable = %f", &itmp.x)==1)            { sound_enable=(itmp.x!=0); }
        if(sscanf(buf, "bgm_volume = %f", &ftmp.x)==1)              { bgm_volume=ftmp.x; }
        if(sscanf(buf, "se_volume = %f", &ftmp.x)==1)               { se_volume=ftmp.x; }
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
    fprintf(f, "bg_multiresolution = %d\n",     bg_multiresolution);
    fprintf(f, "light_multiresolution = %d\n",  light_multiresolution);
    fprintf(f, "show_text = %d\n",              show_text);
    fprintf(f, "show_bloodstain = %d\n",        show_bloodstain);
    fprintf(f, "output_replay = %d\n",          output_replay);
    fprintf(f, "debug_show_grid = %d\n",        debug_show_grid);
    fprintf(f, "debug_show_distance = %d\n",    debug_show_distance);
    fprintf(f, "debug_show_resolution = %d\n",  debug_show_resolution);
    fprintf(f, "sound_enable = %d\n",           sound_enable);
    fprintf(f, "bgm_volume = %f\n",             bgm_volume);
    fprintf(f, "se_volume = %f\n",              se_volume);
    fclose(f);
    return true;
}

void AtomicConfig::setup()
{
    atomicDbgAddParamNodeP("Config/VSync",                bool, &vsync);
    atomicDbgAddParamNodeP("Config/Unlimit Game Speed",   bool, &unlimit_gamespeed);
    atomicDbgAddParamNodeP("Config/PostEffect Bloom",     bool, &posteffect_bloom);
    atomicDbgAddParamNodeP("Config/PostEffect Antialias", bool, &posteffect_antialias);
}


AtomicApplication *g_appinst = NULL;

AtomicApplication* AtomicApplication::getInstance() { return g_appinst; }

AtomicApplication::AtomicApplication()
    : m_request_exit(false)
    , m_game(NULL)
{
    istAssert(g_appinst==NULL);
    g_appinst = this;

#ifdef atomic_enable_debug_log
    m_log = fopen("atomic.log", "wb");
#endif // atomic_enable_debug_log
}

AtomicApplication::~AtomicApplication()
{
#ifdef atomic_enable_debug_log
    if(m_log!=NULL) {
        fclose(m_log);
    }
#endif // atomic_enable_debug_log
    if(g_appinst==this) { g_appinst=NULL; }
}

bool AtomicApplication::initialize(int argc, char *argv[])
{
#ifdef atomic_enable_shader_live_edit
    ::AllocConsole();
#endif // atomic_enable_shader_live_edit
    istTaskSchedulerInitialize();

    // initialize debug menu
    atomicDbgInitializeDebugMenu();

    // console
    istCommandlineInitialize();
    istCommandlineConsoleInitialize();

    m_config.setup();
    m_config.readFromFile(ATOMIC_CONFIG_FILE_PATH);
    if(m_config.window_pos.x >= 30000) { m_config.window_pos.x = 0; }
    if(m_config.window_pos.y >= 30000) { m_config.window_pos.y = 0; }
    if(m_config.window_size.x < 320 || m_config.window_size.x < 240) { m_config.window_size = ivec2(1024, 768); }

    // create window
    ivec2 wpos = m_config.window_pos;
    ivec2 wsize = m_config.window_size;
    if(!super::initialize(wpos, wsize, L"atomic", m_config.fullscreen))
    {
        return false;
    }

    // start rendering thread
    AtomicRenderingSystem::initializeInstance();

    // initialize sound
    AtomicSound::initializeInstance();

    // create game
    m_game = istNew(AtomicGame)();
    if(argc > 1) {
        m_game->readReplayFromFile(argv[1]);
    }

    // start server
    Poco::ThreadPool::defaultPool().addCapacity(8);
    atomicGameServerInitialize();
    atomicGameClientInitialize();
    atomicLevelEditorServerInitialize();

    atomicGameClientConnect("localhost", atomic_GameServer_DefaultPort);

    registerCommands();

    return true;
}

void AtomicApplication::finalize()
{
    m_config.writeToFile(ATOMIC_CONFIG_FILE_PATH);

    atomicLevelEditorServerFinalize();
    atomicGameClientFinalize();
    atomicGameServerFinalize();
    Poco::ThreadPool::defaultPool().joinAll();

    istSafeDelete(m_game);

    AtomicRenderingSystem::finalizeInstance();
    AtomicSound::finalizeInstance();

    istCommandlineConsoleFinalize();
    istCommandlineFinalize();
    atomicDbgFinalizeDebugMenu();

    istTaskSchedulerFinalize();
    FinalizeText();
    ist::PoolNewManager::freeAll();
    FinalizeCrashReporter();
    super::finalize();
}

void AtomicApplication::mainLoop()
{
#ifdef _WIN64
#   define MSBUILD_OPTION "atomic.vcxproj /m /p:Configuration=Release;Platform=x64 /t:ClCompile"
#   define BUILD_TARGET "x64\\Release"
#else // _WIN64
#   define MSBUILD_OPTION "atomic.vcxproj /m /p:Configuration=Release;Platform=Win32 /t:ClCompile"
#   define BUILD_TARGET "Release"
#endif // _WIN64
    DOL_AddSourceDirectory("Game\\Entity");
    DOL_StartAutoRecompile(MSBUILD_OPTION, true);
    DOL_Load(BUILD_TARGET);
    DOL_Link();

    ist::Timer pc;
    const float32 delay = 16.666f;
    const float32 dt = 1.0f;

    while(!m_request_exit)
    {
        DOL_Update();
        translateMessage();
        sysUpdate();

        if(m_game) {
            m_game->frameBegin();
            m_game->update(dt);
            m_game->asyncupdateBegin(dt);
            updateInput();
            m_game->draw();
            m_game->asyncupdateEnd();
            m_game->frameEnd();

            if( m_game->IsWaitVSyncRequired() &&
                (!atomicGetConfig()->unlimit_gamespeed && !atomicGetConfig()->vsync))
            {
                float32 remain = delay-pc.getElapsedMillisec();
                ist::Thread::microSleep((uint32)std::max<float32>(remain*1000.0f, 0.0f));
                pc.reset();
            }
        }
    }
}

void AtomicApplication::sysUpdate()
{
    if(getKeyboardState().isKeyTriggered(ist::KEY_F2)) {
        m_config.posteffect_bloom = !m_config.posteffect_bloom;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F3)) {
        m_config.debug_show_gbuffer--;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F4)) {
        m_config.debug_show_gbuffer++;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F5)) {
        m_config.debug_show_lights--;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F6)) {
        m_config.debug_show_lights++;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F7)) {
        m_config.pause = !m_config.pause;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F8)) {
        m_config.bg_multiresolution = !m_config.bg_multiresolution;
    }
    if(getKeyboardState().isKeyTriggered(ist::KEY_F9)) {
        m_config.debug_show_resolution = !m_config.debug_show_resolution;
    }
    if(getKeyboardState().isKeyTriggered('7')) {
        atomicGetRenderStates()->ShowMultiresolution = !atomicGetRenderStates()->ShowMultiresolution;
    }
    if(getKeyboardState().isKeyPressed('8')) {
        float &p = atomicGetLights()->getMultiresolutionParams().Threshold.x;
        p = clamp(p-0.001f, 0.0f, 1.0f);
    }
    if(getKeyboardState().isKeyPressed('9')) {
        float &p = atomicGetLights()->getMultiresolutionParams().Threshold.x;
        p = clamp(p+0.001f, 0.0f, 1.0f);
    }

    atomicDbgDebugMenuUpdate();
}


void AtomicApplication::updateInput()
{
    super::updateInput();

    RepMove move;
    int buttons = getJoyState().getButtons();

    const ist::MouseState &mouse = getMouseState();
    if((mouse.getButtonState() & ist::MouseState::BU_LEFT)!=0)  { buttons = buttons |= 1<<0; }
    if((mouse.getButtonState() & ist::MouseState::BU_RIGHT)!=0) { buttons = buttons |= 1<<1; }
    if((mouse.getButtonState() & ist::MouseState::BU_MIDDLE)!=0){ buttons = buttons |= 1<<2; }

    const ist::KeyboardState &kb = getKeyboardState();
    if(kb.isKeyPressed('Z')){ buttons = buttons |= 1<<0; }
    if(kb.isKeyPressed('X')){ buttons = buttons |= 1<<1; }
    if(kb.isKeyPressed('C')){ buttons = buttons |= 1<<2; }
    if(kb.isKeyPressed('V')){ buttons = buttons |= 1<<3; }
    if(kb.isKeyPressed(ist::KEY_RIGHT)  || kb.isKeyPressed('D')){ move.x = INT16_MAX; }
    if(kb.isKeyPressed(ist::KEY_LEFT)   || kb.isKeyPressed('A')){ move.x =-INT16_MAX; } // INT16_MIN じゃないのは意図的
    if(kb.isKeyPressed(ist::KEY_UP)     || kb.isKeyPressed('W')){ move.y = INT16_MAX; }
    if(kb.isKeyPressed(ist::KEY_DOWN)   || kb.isKeyPressed('S')){ move.y =-INT16_MAX; }
    if(kb.isKeyTriggered(ist::KEY_F1)) {
        m_config.posteffect_antialias = !m_config.posteffect_antialias;
    }

    {
        RepMove jpos(getJoyState().getX(), -getJoyState().getY());
        if(glm::length(jpos.toF())>0.4f) { move=jpos; }
    }
    m_inputs.update(RepInput(move, buttons));
}

bool AtomicApplication::handleWindowMessage(const ist::WindowMessage& wm)
{
    switch(wm.type)
    {
    case ist::WindowMessage::MES_CLOSE:
        {
            m_request_exit = true;
        }
        return true;

    case ist::WindowMessage::MES_KEYBOARD:
        {
            const ist::WM_Keyboard& m = static_cast<const ist::WM_Keyboard&>(wm);
            if(m.action==ist::WM_Keyboard::ACT_KEYUP && m.key==ist::KEY_ESCAPE) {
                m_request_exit = true;
            }
        }
        return true;

    case ist::WindowMessage::MES_WINDOW_SIZE:
        {
            const ist::WM_WindowSize& m = static_cast<const ist::WM_WindowSize&>(wm);
            m_config.window_size = m.window_size;
        }
        return true;

    case ist::WindowMessage::MES_WINDOW_MOVE:
        {
            const ist::WM_WindowMove& m = static_cast<const ist::WM_WindowMove&>(wm);
            m_config.window_pos = m.window_pos;
        }
        return true;

    case ist::WindowMessage::MES_IME_RESULT:
        {
            const ist::WM_IME& m = static_cast<const ist::WM_IME&>(wm);
            stl::wstring str(m.text, m.text_len);
        }
        return true;

    case ist::WindowMessage::MES_IME_BEGIN:
        {
            istPrint(L"MES_IME_BEGIN\n");
        }
        break;
    case ist::WindowMessage::MES_IME_END:
        {
            istPrint(L"MES_IME_END\n");
        }
        break;
    case ist::WindowMessage::MES_IME_CHAR:
        {
            const ist::WM_IME& m = static_cast<const ist::WM_IME&>(wm);
            stl::wstring str(m.text, m.text_len);
            istPrint(L"MES_IME_CHAR %s\n", str.c_str());
        }
        break;
    }

    return false;
}

void AtomicApplication::handleError(ATOMIC_ERROR e)
{
    stl::wstring mes;
    switch(e) {
    case ATERR_OPENGL_330_IS_NOT_SUPPORTED:   mes=GetText(TID_OPENGL330_IS_NOT_SUPPORTED); break;
    case ATERR_CUDA_NO_DEVICE:                mes=GetText(TID_ERROR_CUDA_NO_DEVICE); break;
    case ATERR_CUDA_INSUFFICIENT_DRIVER:      mes=GetText(TID_ERROR_CUDA_INSUFFICIENT_DRIVER); break;
    }
    istShowMessageDialog(mes.c_str(), L"error", DLG_OK);
}


void AtomicApplication::drawCallback()
{
    m_game->drawCallback();
}



#ifdef atomic_enable_debug_log
void AtomicApplication::printDebugLog( const char *format, ... )
{
    if(m_log==NULL) { return; }
    va_list vl;
    va_start(vl, format);
    fprintf(m_log, "%d ", (uint32)atomicGetFrame());
    vfprintf(m_log, format, vl);
    va_end(vl);
}

void AtomicApplication::registerCommands()
{
}

#endif // atomic_enable_debug_log


} // namespace atomic