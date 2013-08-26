#ifndef atm_Engine_Graphics_AtomicRenderingSystem_h
#define atm_Engine_Graphics_AtomicRenderingSystem_h

#include "psym/psym.h"

namespace atm {

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

} // namespace atm

#define atmGetRenderingSystem()              AtomicRenderingSystem::getInstance()
#define atmWaitUntilDrawCallbackComplete()   atmGetRenderingSystem()->waitUntilDrawCallbackComplete()
#define atmKickDraw()                        atmGetRenderingSystem()->kickDraw()
#define atmGetAverageFPS()                   atmGetRenderingSystem()->getAverageFPS()
#define atmGetGLDevice()                     atmGetRenderingSystem()->getDevice()
#define atmGetGLDeviceContext()              atmGetRenderingSystem()->getDeviceContext()
#define atmGetEasyDrawer()                   atmGetRenderingSystem()->getEasyDrawer()

#endif // atm_Engine_Graphics_AtomicRenderingSystem_h
