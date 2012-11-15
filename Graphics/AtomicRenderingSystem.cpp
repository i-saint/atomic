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

class AtomicRenderingThread : public ist::Thread
{
private:
    i3d::Device *m_device;
    i3d::DeviceContext *m_context;

    ATOMIC_ERROR m_error;
    ist::Mutex m_mutex_request;
    ist::Condition m_cond_request;
    ist::Condition m_cond_initialize_complete;
    ist::Condition m_cond_callback_complete;
    stl::vector<RenderingRequest> m_requests;
    stl::vector<RenderingRequest> m_requests_temp;

    PerformanceCounter m_fps_counter;

public:
    AtomicRenderingThread();
    ~AtomicRenderingThread();

    void exec();

    void pushRequest(const RenderingRequest &req);
    RenderingRequest popRequest();

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
    , m_error(ATERR_NOERROR)
{
}

AtomicRenderingThread::~AtomicRenderingThread()
{
    pushRequest(RenderingRequest::createStopRequest());
    join();
}

void AtomicRenderingThread::waitUntilInitializationComplete()
{
    m_cond_initialize_complete.wait();
}

void AtomicRenderingThread::waitUntilDrawCallbackComplete()
{
    m_cond_callback_complete.wait();
}

void AtomicRenderingThread::pushRequest(const RenderingRequest &req)
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_request);
        m_requests.push_back(req);
    }
    m_cond_request.signalOne();
}

void AtomicRenderingThread::exec()
{
    ist::Thread::setNameToCurrentThread("AtomicRenderingThread");

    m_device = i3d::CreateDevice(atomicGetApplication()->getWindowHandle());
    if(!GLEW_VERSION_3_3) {
        m_error = ATERR_OPENGL_330_IS_NOT_SUPPORTED;
        m_cond_initialize_complete.signalOne();
        goto finalize_section;
    }
    m_context = m_device->createContext();

#ifdef ist_env_Windows
    wglSwapIntervalEXT(atomicGetConfig()->vsync);
#endif // ist_env_Windows
    GraphicResourceManager::intializeInstance();
    AtomicRenderer::initializeInstance();
    m_cond_initialize_complete.signalOne();

    bool end_flag = false;
    while(!end_flag) {
        m_cond_request.wait();
        {
            ist::Mutex::ScopedLock lock(m_mutex_request);
            m_requests_temp = m_requests;
            m_requests.clear();
        }
        for(size_t i=0; i<m_requests_temp.size(); ++i) {
            RenderingRequest req = m_requests_temp[i];
            switch(req.type) {
            case RenderingRequest::REQ_STOP: end_flag=true; break;
            case RenderingRequest::REQ_RENDER: doRender(); break;
            }
        }
        m_requests_temp.clear();
    }

    AtomicRenderer::finalizeInstance();
    GraphicResourceManager::finalizeInstance();

finalize_section:
    istSafeRelease(m_context);
#ifdef i3d_enable_resource_leak_check
    m_device->printLeakInfo();
#endif // __i3d_enable_leak_check__
    istSafeRelease(m_device);
}

void AtomicRenderingThread::doRender()
{
    atomicGetApplication()->drawCallback();
    m_cond_callback_complete.signalOne();
    m_device->swapBuffers();
    m_fps_counter.count();
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
