#include "stdafx.h"
#include "types.h"
#include "AtomicGame.h"
#include "AtomicApplication.h"
#include "Text.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/World.h"
#include "Sound/AtomicSound.h"

#define ATOMIC_CONFIG_FILE_PATH "atomic.conf"

namespace atomic {


class AtomicRenderingThread
{
private:
    AtomicApplication *m_app;
    boost::scoped_ptr<boost::thread> m_thread;
    boost::condition_variable m_cond_wait_for_initialize;
    boost::condition_variable m_cond_wait_for_draw;
    boost::condition_variable m_cond_wait_for_complete;
    boost::condition_variable m_cond_wait_for_end;
    boost::mutex m_mutex_wait_for_initialize;
    boost::mutex m_mutex_wait_for_draw;
    boost::mutex m_mutex_wait_for_complete;
    boost::mutex m_mutex_wait_for_end;
    bool m_stop_flag;
    bool m_is_initialized;
    bool m_is_ready_to_draw;
    bool m_is_draw_complete;
    bool m_is_end;
    PerformanceCounter m_fps_counter;

public:
    AtomicRenderingThread(AtomicApplication *app);
    ~AtomicRenderingThread();
    void requestStop();
    void operator()();

    void run();
    void stop();
    void waitForInitializeComplete();
    void waitForDrawComplete();
    void kick();

    float32 getAverageFPS() const { return m_fps_counter.getAverageFPS(); }
};

AtomicRenderingThread::AtomicRenderingThread(AtomicApplication *app)
: m_app(app)
, m_stop_flag(false)
, m_is_initialized(false)
, m_is_ready_to_draw(false)
, m_is_draw_complete(true)
, m_is_end(false)
{
}

AtomicRenderingThread::~AtomicRenderingThread()
{
    if(m_thread) {
        m_thread->join();
    }
}

void AtomicRenderingThread::run()
{
    m_thread.reset(new boost::thread(boost::ref(*this)));
}

void AtomicRenderingThread::stop()
{
    m_stop_flag = true;
    kick();

    boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_end);
    while(!m_is_end) {
        m_cond_wait_for_end.wait(lock);
    }
}

void AtomicRenderingThread::waitForInitializeComplete()
{
    boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_initialize);
    while(!m_is_initialized) {
        m_cond_wait_for_initialize.wait(lock);
    }
}

void AtomicRenderingThread::waitForDrawComplete()
{
    boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_complete);
    while(!m_is_draw_complete) {
        m_cond_wait_for_complete.wait(lock);
    }
}

void AtomicRenderingThread::kick()
{
    waitForDrawComplete();
    {
        boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_draw);
        m_is_ready_to_draw = true;
        m_is_draw_complete = false;
        m_cond_wait_for_draw.notify_all();
    }
}


void AtomicRenderingThread::operator()()
{
    ist::SetThreadName("AtomicRenderingThread");

    bool initialized = m_app->initializeDraw();
    if(initialized) {
        GraphicResourceManager::intializeInstance();
        AtomicRenderer::initializeInstance();
    }
    {
        boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_initialize);
        m_is_initialized = true;
        m_cond_wait_for_initialize.notify_all();
    }
    if(!initialized) { goto APP_END; }

    {
        boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_draw);
        while(!m_stop_flag) {
            while(!m_is_ready_to_draw) {
                m_cond_wait_for_draw.wait(lock);
            }
            m_is_ready_to_draw = false;
            atomicGetApplication()->drawCallback();
            {
                boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_complete);
                m_is_draw_complete = true;
                m_cond_wait_for_complete.notify_all();
            }

            m_fps_counter.count();
        }
    }

    AtomicRenderer::finalizeInstance();
    GraphicResourceManager::finalizeInstance();

APP_END:
    m_app->finalizeDraw();
    m_is_end = true;
    m_cond_wait_for_end.notify_all();
}





AtomicConfig::AtomicConfig()
{
    window_pos      = ivec2(0, 0);
    window_size     = ivec2(1024, 768);
    fullscreen      = false;
    sound_volume    = 0.5;
    posteffect_antialias    = true;
    posteffect_bloom        = true;
    posteffect_motionblur   = true;

}

bool AtomicConfig::readFromFile( const char* filepath )
{
    if(FILE *f=fopen(filepath, "r")) {
        char buf[256];
        while(fgets(buf, 256, f)) {
            ivec2 itmp;
            vec2 ftmp;
            if(sscanf(buf, "window_pos = %d, %d", &itmp.x, &itmp.y)==2) { window_pos.x=itmp.x; window_pos.y=itmp.y; }
            if(sscanf(buf, "window_size = %d, %d", &itmp.x, &itmp.y)==2){ window_size.x=itmp.x; window_size.y=itmp.y; }
            if(sscanf(buf, "sound_volume = %f", &ftmp.x)==1)            { sound_volume=ftmp.x; }
            if(sscanf(buf, "posteffect_antialias = %d", &itmp.x)==1)    { posteffect_antialias=(itmp.x!=0); }
            if(sscanf(buf, "posteffect_bloom = %d", &itmp.x)==1)        { posteffect_bloom=(itmp.x!=0); }
            if(sscanf(buf, "posteffect_motionblur = %d", &itmp.x)==1)   { posteffect_motionblur=(itmp.x!=0); }
        }
        fclose(f);
        return true;
    }
    return false;
}

bool AtomicConfig::writeToFile( const char* filepath )
{
    if(FILE *f=fopen(filepath, "w")) {
        fprintf(f, "window_pos = %d, %d\n", window_pos.x, window_pos.y);
        fprintf(f, "window_size = %d, %d\n", window_size.x, window_size.y);
        fprintf(f, "fullscreen = %d\n", fullscreen);
        fprintf(f, "sound_volume = %f\n", sound_volume);
        fprintf(f, "posteffect_antialias = %d\n", posteffect_antialias);
        fprintf(f, "posteffect_bloom = %d\n", posteffect_bloom);
        fprintf(f, "posteffect_motionblur = %d\n", posteffect_motionblur);
        fclose(f);
        return true;
    }
    return false;
}


AtomicApplication *g_appinst = NULL;

AtomicApplication* AtomicApplication::getInstance() { return g_appinst; }

AtomicApplication::AtomicApplication()
    : m_request_exit(false)
    , m_game(NULL)
    , m_renderng_thread(NULL)
    , m_sound_thread(NULL)
{
    if(g_appinst) { IST_ASSERT("already initialized"); }
    g_appinst = this;

}

AtomicApplication::~AtomicApplication()
{
    if(g_appinst==this) { g_appinst=NULL; }
}

bool AtomicApplication::initialize()
{
    InitializeText();
    m_config.readFromFile(ATOMIC_CONFIG_FILE_PATH);
#ifndef ATOMIC_ENABLE_DEBUG_FEATURE
    {
        ist::DisplaySetting ds = getCurrentDisplaySetting();
        if(m_config.window_pos.x >= ds.getResolution().x) { m_config.window_pos.x = 0; }
        if(m_config.window_pos.y >= ds.getResolution().y) { m_config.window_pos.y = 0; }
    }
#endif // ATOMIC_ENABLE_DEBUG_FEATURE

    ivec2 wpos = m_config.window_pos;
    ivec2 wsize = m_config.window_size;
    if(!super::initialize(wpos, wsize, L"atomic", m_config.fullscreen))
    {
        return false;
    }
    TaskScheduler::initializeSingleton();
    //TaskScheduler::initializeSingleton(11);

    m_renderng_thread = IST_NEW16(AtomicRenderingThread)(this);
    m_renderng_thread->run();
    m_renderng_thread->waitForInitializeComplete();

    {
        ERROR_CODE e = getGraphicsError();
        if(!GLEW_VERSION_3_3) { e=ERR_OPENGL_330_IS_NOT_SUPPORTED; }
        if(e!=ERR_NOERROR) {
            handleError(e);
            IST_SAFE_DELETE(m_renderng_thread);
            return false;
        }
    }

    m_sound_thread = IST_NEW16(AtomicSoundThread)();
    m_sound_thread->run();

    m_game = IST_NEW16(AtomicGame)();

    return true;
}

void AtomicApplication::finalize()
{
    if(m_renderng_thread) { m_renderng_thread->stop(); }
    if(m_sound_thread) { m_sound_thread->requestStop(); }
    IST_SAFE_DELETE(m_game);
    IST_SAFE_DELETE(m_sound_thread);
    IST_SAFE_DELETE(m_renderng_thread);

    TaskScheduler::finalizeSingleton();
    super::finalize();

    m_config.writeToFile(ATOMIC_CONFIG_FILE_PATH);
    FinalizeText();
}

void AtomicApplication::mainLoop()
{
    while(!m_request_exit)
    {
        translateMessage();
        updateInput();

        PerformanceCounter pc;
        float dt = 0.0f;
        if(m_game) {
            m_game->update(dt);
            m_game->draw();
            dt = pc.getElapsedMillisecond();
            pc.reset();
        }
    }
}

void AtomicApplication::updateInput()
{
    super::updateInput();

    vec2 move = vec2(0.0f);
    int buttons = getJoyState().getButtons();
    if(getKeyboardState().isKeyPressed(VK_RIGHT))   { move.x = 1.0f; }
    if(getKeyboardState().isKeyPressed(VK_LEFT))    { move.x =-1.0f; }
    if(getKeyboardState().isKeyPressed(VK_UP))      { move.y = 1.0f; }
    if(getKeyboardState().isKeyPressed(VK_DOWN))    { move.y =-1.0f; }
    {
        vec2 jpos = vec2((float)getJoyState().getX(), -(float)getJoyState().getY());
        jpos /= 32768.0f;
        if(glm::length(jpos)>0.4f) { move=jpos; }
    }
    m_inputs.setMove(move);
    m_inputs.setButtons(buttons);
}

int AtomicApplication::handleWindowMessage(const ist::WindowMessage& wm)
{
    switch(wm.type)
    {
    case ist::WindowMessage::MES_CLOSE:
        {
            m_request_exit = true;
        }
        return 0;

    case ist::WindowMessage::MES_KEYBOARD:
        {
            const ist::WM_Keyboard& m = static_cast<const ist::WM_Keyboard&>(wm);
            if(m.action==ist::WM_Keyboard::ACT_KEYUP && m.key==ist::WM_Keyboard::KEY_ESCAPE) {
                m_request_exit = true;
            }
        }
        return 0;

    case ist::WindowMessage::MES_WINDOW_SIZE:
        {
            const ist::WM_WindowSize& m = static_cast<const ist::WM_WindowSize&>(wm);
            m_config.window_size = m.window_size;
        }
        return 0;

    case ist::WindowMessage::MES_WINDOW_MOVE:
        {
            const ist::WM_WindowMove& m = static_cast<const ist::WM_WindowMove&>(wm);
            m_config.window_pos = m.window_pos;
        }
        return 0;

    case ist::WindowMessage::MES_IME_RESULT:
        {
            const ist::WM_IME& m = static_cast<const ist::WM_IME&>(wm);
            handleCommandLine(m.text, m.text_len);
        }
        return 0;

    case ist::WindowMessage::MES_IME_BEGIN: IST_PRINT(L"MES_IME_BEGIN\n"); break;
    case ist::WindowMessage::MES_IME_END: IST_PRINT(L"MES_IME_END\n"); break;
    case ist::WindowMessage::MES_IME_CHAR: IST_PRINT(L"MES_IME_CHAR\n"); break;
    }

    return 0;
}

void AtomicApplication::handleError(ERROR_CODE e)
{
    std::wstring mes;
    switch(e) {
    case ERR_OPENGL_330_IS_NOT_SUPPORTED:   mes=GetText(TID_OPENGL330_IS_NOT_SUPPORTED); break;
    case ERR_CUDA_NO_DEVICE:                mes=GetText(TID_ERROR_CUDA_NO_DEVICE); break;
    case ERR_CUDA_INSUFFICIENT_DRIVER:      mes=GetText(TID_ERROR_CUDA_INSUFFICIENT_DRIVER); break;
    }
    istShowMessageDialog(mes.c_str(), L"error", DLG_OK);
}

void AtomicApplication::handleCommandLine( const wchar_t* command, size_t command_len )
{
    IST_PRINT(L"%s\n", command);
}


void AtomicApplication::waitForDrawComplete()
{
    m_renderng_thread->waitForDrawComplete();
}

void AtomicApplication::kickDraw()
{
    m_renderng_thread->kick();
}

void AtomicApplication::drawCallback()
{
    m_game->drawCallback();
}

atomic::float32 AtomicApplication::getAverageFPS() const
{
    if(m_renderng_thread) { return m_renderng_thread->getAverageFPS(); }
    return 0.0f;
}


} // namespace atomic