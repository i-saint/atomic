#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "AtomicRenderingSystem.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"

namespace atomic {

struct RenderingRequest
{
    enum REQ_TYPE {
        REQ_UNKNOWN,
        REQ_RENDER,
        REQ_STOP,
        REQ_VSYNC,
    };
    REQ_TYPE type;

    static RenderingRequest createRenderRequest()
    {
        RenderingRequest r = {REQ_RENDER};
        return r;
    }

    static RenderingRequest createStopRequest()
    {
        RenderingRequest r = {REQ_STOP};
        return r;
    }
};

class AtomicRenderingThread
{
public:
    enum STATE {
        ST_BEFORE_INITIALIZE,
        ST_IDLING,
        ST_BUSY,
        ST_PROCESSING_CALLBACK,
        ST_PROCESSING_GL,
        ST_ERROR,
    };

private:
    i3d::Device *m_device;
    i3d::DeviceContext *m_context;
    boost::thread *m_thread;

    ATOMIC_ERROR m_error;
    volatile uint32 m_state;
    boost::mutex m_mutex_request;
    boost::condition_variable m_cond_request;
    stl::vector<RenderingRequest> m_requests;

    PerformanceCounter m_fps_counter;

public:
    AtomicRenderingThread();
    ~AtomicRenderingThread();
    void operator()();
    void run();

    void pushRequest(const RenderingRequest &req);
    RenderingRequest popRequest();

    void setState(STATE v);
    STATE getState() const;
    void waitUntilInitializationComplete();
    void waitUntilDrawCallbackComplete();

    void doRender();

    ATOMIC_ERROR getError() const { return m_error; }
    float32 getAverageFPS() const { return m_fps_counter.getAverageFPS(); }
    i3d::Device* getDevice() { return m_device; }
    i3d::DeviceContext* getDeviceContext() { return m_context; }
};

AtomicRenderingThread::AtomicRenderingThread()
    : m_device(NULL)
    , m_context(NULL)
    , m_thread(NULL)
    , m_error(ATERR_NOERROR)
    , m_state(ST_BEFORE_INITIALIZE)
{
}

AtomicRenderingThread::~AtomicRenderingThread()
{
    if(m_thread) {
        pushRequest(RenderingRequest::createStopRequest());
        m_thread->join();
        istSafeDelete(m_thread);
    }
}

void AtomicRenderingThread::run()
{
    m_thread = istNew( boost::thread(boost::ref(*this)) );
}

void AtomicRenderingThread::waitUntilInitializationComplete()
{
    for(;;) {
        STATE r = getState();
        if(r!=ST_BEFORE_INITIALIZE) { break; }
        ::Sleep(0);
    }
}

void AtomicRenderingThread::waitUntilDrawCallbackComplete()
{
    for(;;) {
        STATE r = getState();
        if(r!=ST_BUSY && r!=ST_PROCESSING_CALLBACK) { break; }
        ::Sleep(1);
    }
}

void AtomicRenderingThread::pushRequest(const RenderingRequest &req)
{
    boost::lock_guard<boost::mutex> lock(m_mutex_request);
    m_requests.push_back(req);
    setState(ST_BUSY);
    m_cond_request.notify_all();
}

RenderingRequest AtomicRenderingThread::popRequest()
{
    boost::unique_lock<boost::mutex> lock(m_mutex_request);
    if(m_requests.empty()) {
        setState(ST_IDLING);
        m_cond_request.wait(lock);
    }
    RenderingRequest r = m_requests.back();
    m_requests.pop_back();
    return r;
}


void AtomicRenderingThread::operator()()
{
    ist::SetThreadName("AtomicRenderingThread");

    m_device = istNew(i3d::Device)(atomicGetApplication()->getWindowHandle());
    if(!GLEW_VERSION_3_3) {
        m_error = ATERR_OPENGL_330_IS_NOT_SUPPORTED;
        goto finalize_section;
    }

    wglSwapIntervalEXT(atomicGetConfig()->vsync);
    GraphicResourceManager::intializeInstance();
    AtomicRenderer::initializeInstance();

    for(;;) {
        RenderingRequest req = popRequest();
        switch(req.type) {
        case RenderingRequest::REQ_STOP: goto end_section; break;
        case RenderingRequest::REQ_RENDER: doRender(); break;
        }
    }

end_section:
    AtomicRenderer::finalizeInstance();
    GraphicResourceManager::finalizeInstance();

finalize_section:
    istSafeDelete(m_device);
}

void AtomicRenderingThread::doRender()
{
    setState(ST_PROCESSING_CALLBACK);
    atomicGetApplication()->drawCallback();

    setState(ST_PROCESSING_GL);
    m_device->swapBuffers();

    m_fps_counter.count();
}

void AtomicRenderingThread::setState( STATE v )
{
    InterlockedExchange(&m_state, v);
}

AtomicRenderingThread::STATE AtomicRenderingThread::getState() const
{
    MemoryBarrier();
    STATE r = (STATE)m_state;
    MemoryBarrier();
    return r;
}





AtomicRenderingSystem * AtomicRenderingSystem::s_inst = NULL;

bool AtomicRenderingSystem::initializeInstance()
{
    if(!s_inst) {
        s_inst = istNew(AtomicRenderingSystem)();
        s_inst->waitUntilInitializationComplete();
        return true;
    }
    return false;
}

void AtomicRenderingSystem::finalizeInstance()
{
    istSafeDelete(s_inst);
}

AtomicRenderingSystem* AtomicRenderingSystem::getInstance()
{
    return s_inst;
}

AtomicRenderingSystem::AtomicRenderingSystem()
    : m_render_thread(NULL)
{
    m_render_thread = istNew(AtomicRenderingThread)();
    m_render_thread->run();
}

AtomicRenderingSystem::~AtomicRenderingSystem()
{
    istSafeDelete(m_render_thread);
}

void AtomicRenderingSystem::waitUntilInitializationComplete()
{
    m_render_thread->waitUntilInitializationComplete();
}

void AtomicRenderingSystem::waitUntilDrawCallbackComplete()
{
    m_render_thread->waitUntilDrawCallbackComplete();
}

void AtomicRenderingSystem::kickDraw()
{
    m_render_thread->pushRequest(RenderingRequest::createRenderRequest());
}

float32 AtomicRenderingSystem::getAverageFPS() const
{
    return m_render_thread->getAverageFPS();
}

i3d::Device * AtomicRenderingSystem::getDevice()
{
    return m_render_thread->getDevice();
}

i3d::DeviceContext * AtomicRenderingSystem::getDeviceContext()
{
    return m_render_thread->getDeviceContext();
}

} // namespace atomic
