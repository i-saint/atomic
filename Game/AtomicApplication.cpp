#include "stdafx.h"
#include "../types.h"
#include "World.h"
#include "AtomicGame.h"
#include "AtomicApplication.h"
#include "../Graphics/GraphicResourceManager.h"
#include "../Graphics/Renderer.h"


namespace atomic {


class AtomicDrawThread
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

public:
    AtomicDrawThread(AtomicApplication *app);
    ~AtomicDrawThread();
    void requestStop();
    void operator()();

    void run();
    void stop();
    void waitForInitializeComplete();
    void waitForDrawComplete();
    void kick();
};

AtomicDrawThread::AtomicDrawThread(AtomicApplication *app)
: m_app(app)
, m_stop_flag(false)
, m_is_initialized(false)
, m_is_ready_to_draw(false)
, m_is_draw_complete(true)
, m_is_end(false)
{
}

AtomicDrawThread::~AtomicDrawThread()
{
}

void AtomicDrawThread::run()
{
    m_thread.reset(new boost::thread(boost::ref(*this)));
}

void AtomicDrawThread::stop()
{
    m_stop_flag = true;
    kick();

    boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_end);
    while(!m_is_end) {
        m_cond_wait_for_end.wait(lock);
    }
}

void AtomicDrawThread::waitForInitializeComplete()
{
    boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_initialize);
    while(!m_is_initialized) {
        m_cond_wait_for_initialize.wait(lock);
    }
}

void AtomicDrawThread::waitForDrawComplete()
{
    boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_complete);
    while(!m_is_draw_complete) {
        m_cond_wait_for_complete.wait(lock);
    }
}

void AtomicDrawThread::kick()
{
    waitForDrawComplete();
    m_is_ready_to_draw = true;
    m_is_draw_complete = false;
    m_cond_wait_for_draw.notify_all();
}

void AtomicDrawThread::operator()()
{
    m_app->initializeDraw();
    GraphicResourceManager::intializeInstance();
    m_is_initialized = true;
    m_cond_wait_for_initialize.notify_all();

    {
        boost::unique_lock<boost::mutex> lock(m_mutex_wait_for_draw);
        while(!m_stop_flag) {
            while(!m_is_ready_to_draw) {
                m_cond_wait_for_draw.wait(lock);
            }
            m_is_ready_to_draw = false;
            boost::timer t;
            AtomicRenderer::getInstance()->draw();
            //IST_PRINT("%lf\n", t.elapsed());
            m_is_draw_complete = true;
            m_cond_wait_for_complete.notify_all();
        }
    }
    GraphicResourceManager::finalizeInstance();
    m_app->finalizeDraw();
    m_is_end = true;
    m_cond_wait_for_end.notify_all();
}



static AtomicApplication *s_inst = NULL;


AtomicApplication* AtomicApplication::getInstance() { return s_inst; }

AtomicApplication::AtomicApplication()
    : m_request_exit(false)
    , m_game(NULL)
{
    if(s_inst) {
        IST_ASSERT("already initialized");
    }
    s_inst = this;
}

AtomicApplication::~AtomicApplication()
{
    if(s_inst==this) {
        s_inst = NULL;
    }
}

bool AtomicApplication::initialize(size_t x, size_t y, size_t width, size_t height, const wchar_t *title, bool fullscreen)
{
    if(!super::initialize(x,y, width, height, title, fullscreen))
    {
        return false;
    }
    TaskScheduler::initializeSingleton();

    m_draw_thread = AT_NEW(AtomicDrawThread) AtomicDrawThread(this);
    m_draw_thread->run();
    m_draw_thread->waitForInitializeComplete();

    m_game = AT_ALIGNED_NEW(AtomicGame, 16) AtomicGame();

    return true;
}

void AtomicApplication::finalize()
{
    m_draw_thread->stop();
    AT_DELETE(m_game);
    AT_DELETE(m_draw_thread);

    TaskScheduler::finalizeSingleton();
    super::finalize();
}

void AtomicApplication::mainLoop()
{
    while(!m_request_exit)
    {
        translateMessage();

        if(m_game) {
            m_game->update();
            m_game->draw();
        }
    }
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
    m_draw_thread->waitForDrawComplete();
}

void AtomicApplication::kickDraw()
{
    m_draw_thread->kick();
}


} // namespace atomic