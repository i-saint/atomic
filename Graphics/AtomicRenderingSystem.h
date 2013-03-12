#ifndef atomic_Graphics_AtomicRenderingSystem_h
#define atomic_Graphics_AtomicRenderingSystem_h

#include "psym/psym.h"

namespace atomic {

class AtomicRenderingThread;

class AtomicRenderingSystem
{
public:
private:
    static AtomicRenderingSystem *s_inst;

    AtomicRenderingThread *m_render_thread;

    AtomicRenderingSystem();

public:
    ~AtomicRenderingSystem();

    static bool initializeInstance();
    static void finalizeInstance();
    static AtomicRenderingSystem* getInstance();

    void waitUntilInitializationComplete();
    void waitUntilDrawCallbackComplete();
    void kickDraw();

    uint32 getAverageFPS() const;

    // 以下レンダリングスレッドからのみアクセス可
    i3d::Device* getDevice();
    i3d::DeviceContext* getDeviceContext();
    i3d::EasyDrawer* getEasyDrawer();
};

} // namespace atomic

#define atomicGetRenderingSystem()              AtomicRenderingSystem::getInstance()
#define atomicWaitUntilDrawCallbackComplete()   atomicGetRenderingSystem()->waitUntilDrawCallbackComplete()
#define atomicKickDraw()                        atomicGetRenderingSystem()->kickDraw()
#define atomicGetAverageFPS()                   atomicGetRenderingSystem()->getAverageFPS()
#define atomicGetGLDevice()                     atomicGetRenderingSystem()->getDevice()
#define atomicGetGLDeviceContext()              atomicGetRenderingSystem()->getDeviceContext()
#define atomicGetEasyDrawer()                   atomicGetRenderingSystem()->getEasyDrawer()

#endif // atomic_Graphics_AtomicRenderingSystem_h
