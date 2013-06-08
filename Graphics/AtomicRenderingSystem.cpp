#include "stdafx.h"
#include "ist/iui.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "AtomicRenderingSystem.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"

namespace atm {

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
    i3d::EasyDrawer *m_drawer;

    ATOMIC_ERROR m_error;
    ist::Mutex m_mutex_request;
    ist::Condition m_cond_request;
    ist::Condition m_cond_initialize_complete;
    ist::Condition m_cond_callback_complete;
    ist::vector<RenderingRequest> m_requests;
    ist::vector<RenderingRequest> m_requests_temp;

    ist::Timer m_fps_timer;
    uint32 m_fps_count;
    uint32 m_fps_avg;

public:
    AtomicRenderingThread();
    ~AtomicRenderingThread();

    void exec();

    void pushRequest(const RenderingRequest &req);
    RenderingRequest popRequest();

    void waitUntilInitializationComplete();
    void waitUntilDrawCallbackComplete();

    void doRender();

    ATOMIC_ERROR        getError() const        { return m_error; }
    uint32              getAverageFPS() const   { return m_fps_avg; }
    i3d::Device*        getDevice()             { return m_device; }
    i3d::DeviceContext* getDeviceContext()      { return m_context; }
    i3d::EasyDrawer*    getEasyDrawer()         { return m_drawer; }
};

AtomicRenderingThread::AtomicRenderingThread()
    : m_device(NULL)
    , m_context(NULL)
    , m_drawer(NULL)
    , m_error(ATERR_NOERROR)
    , m_fps_count(0)
    , m_fps_avg(0)
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

    m_device = i3d::CreateDevice(atmGetApplication()->getWindowHandle());
    if(!GLEW_VERSION_3_3) {
        m_error = ATERR_OPENGL_330_IS_NOT_SUPPORTED;
        m_cond_initialize_complete.signalOne();
        goto finalize_section;
    }
    m_context = m_device->createImmediateContext();
    m_drawer = i3d::CreateEasyDrawer();

#ifdef ist_env_Windows
    wglSwapIntervalEXT(atmGetConfig()->vsync);
#endif // ist_env_Windows
    GraphicResourceManager::intializeInstance();
    AtomicRenderer::initializeInstance();
    iuiInitializeRenderer(atmGetEasyDrawer(), atmGetFont());
    m_cond_initialize_complete.signalOne();

    m_fps_timer.reset();
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

        {
            float32 elap = m_fps_timer.getElapsedMillisec();
            if(elap>1000.0f) {
                m_fps_avg = m_fps_count;
                m_fps_count = 0;
                m_fps_timer.reset();
            }
        }
    }

    iuiFinalizeRenderer();
    AtomicRenderer::finalizeInstance();
    GraphicResourceManager::finalizeInstance();

finalize_section:
    istSafeRelease(m_drawer);
    istSafeRelease(m_context);
#ifdef i3d_enable_resource_leak_check
    m_device->printLeakInfo();
#endif // i3d_enable_resource_leak_check
    istSafeRelease(m_device);
}

void AtomicRenderingThread::doRender()
{
    atmGetGraphicsResourceManager()->update();
    atmGetApplication()->drawCallback();
    {
        //static uint32 s_frames;
        //static float32 s_elapsed;
        //i3d::Query_TimeElapsed te;
        //te.begin();

        AtomicRenderer::getInstance()->draw();
        m_cond_callback_complete.signalOne(); // 
        m_device->swapBuffers();

        //s_elapsed += te.end();
        //if(++s_frames==60) {
        //    istPrint("AtomicRenderingThread::doRender(): %.2f\n", s_elapsed/s_frames);
        //    s_elapsed = 0.0f;
        //    s_frames = 0;
        //}
    }
    ++m_fps_count;
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

uint32 AtomicRenderingSystem::getAverageFPS() const
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

i3d::EasyDrawer* AtomicRenderingSystem::getEasyDrawer()
{
    return m_render_thread->getEasyDrawer();
}

} // namespace atm
