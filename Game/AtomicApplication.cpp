#include "stdafx.h"
#include "types.h"
#include "AtomicGame.h"
#include "AtomicApplication.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Game/World.h"
#include "Sound/AtomicSound.h"

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

    m_app->initializeDraw();
    GraphicResourceManager::intializeInstance();
    {
        boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_initialize);
        m_is_initialized = true;
        m_cond_wait_for_initialize.notify_all();
    }

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
    GraphicResourceManager::finalizeInstance();
    m_app->finalizeDraw();
    m_is_end = true;
    m_cond_wait_for_end.notify_all();
}



AtomicApplication *g_appinst = NULL;


AtomicApplication* AtomicApplication::getInstance() { return g_appinst; }

AtomicApplication::AtomicApplication()
    : m_request_exit(false)
    , m_game(NULL)
    , m_renderng_thread(NULL)
    , m_sound_thread(NULL)
{
    if(g_appinst) {
        IST_ASSERT("already initialized");
    }
    g_appinst = this;
}

AtomicApplication::~AtomicApplication()
{
    if(g_appinst==this) { g_appinst=NULL; }
}

bool AtomicApplication::initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen)
{
    if(!super::initialize(x,y, width, height, title, fullscreen))
    {
        return false;
    }
    TaskScheduler::initializeSingleton();
    //TaskScheduler::initializeSingleton(11);

    m_renderng_thread = IST_NEW16(AtomicRenderingThread)(this);
    m_renderng_thread->run();
    m_renderng_thread->waitForInitializeComplete();

    m_sound_thread = IST_NEW16(AtomicSoundThread)();
    m_sound_thread->run();

    m_game = IST_NEW16(AtomicGame)();

    return true;
}

void AtomicApplication::finalize()
{
    m_renderng_thread->stop();
    m_sound_thread->requestStop();
    IST_SAFE_DELETE(m_game);
    IST_SAFE_DELETE(m_sound_thread);
    IST_SAFE_DELETE(m_renderng_thread);

    TaskScheduler::finalizeSingleton();
    super::finalize();
}

void AtomicApplication::mainLoop()
{
    while(!m_request_exit)
    {
        translateMessage();
        updateInput();

        if(m_game) {
            m_game->update();
            m_game->draw();
        }
    }
}

void AtomicApplication::updateInput()
{
    super::updateInput();

    float2 move = make_float2(0.0f);
    int buttons = getJoyState().getButtons();
    if(getKeyboardState().isKeyPressed(VK_RIGHT))   { move.x = 1.0f; }
    if(getKeyboardState().isKeyPressed(VK_LEFT))    { move.x =-1.0f; }
    if(getKeyboardState().isKeyPressed(VK_UP))      { move.y = 1.0f; }
    if(getKeyboardState().isKeyPressed(VK_DOWN))    { move.y =-1.0f; }
    {
        float2 jpos = make_float2((float)getJoyState().getX(), -(float)getJoyState().getY());
        jpos /= 32768.0f;
        if(length(jpos)>0.4f) { move=jpos; }
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
            ist::WM_Keyboard m = (ist::WM_Keyboard&)wm;
            if(m.action==ist::WM_Keyboard::ACT_KEYUP && m.key==ist::WM_Keyboard::KEY_ESCAPE) {
                m_request_exit = true;
            }
        }
    }

    return 0;
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