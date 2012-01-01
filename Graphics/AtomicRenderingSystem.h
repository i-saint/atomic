#ifndef __atomic_Graphics_AtomicGraphics__
#define __atomic_Graphics_AtomicGraphics__


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

    void waitForInitializeComplete();
    void waitForDrawCallbackComplete();
    void kickDraw();

    float32 getAverageFPS() const;

    // 以下レンダリングスレッドからのみアクセス可
    i3d::Device* getDevice();
    i3d::DeviceContext* getDeviceContext();
};

} // namespace atomic

#define atomicGetRenderingSystem()          AtomicRenderingSystem::getInstance()
#define atomicWaitForDrawCallbackComplete() atomicGetRenderingSystem()->waitForDrawCallbackComplete()
#define atomicKickDraw()                    atomicGetRenderingSystem()->kickDraw()
#define atomicGetAverageFPS()               atomicGetRenderingSystem()->getAverageFPS()
#define atomicGetGraphicsDevice()           atomicGetRenderingSystem()->getDevice()
#define atomicGetGraphicsDeviceContext()    atomicGetRenderingSystem()->getDeviceContext()

#endif // __atomic_Graphics_AtomicGraphics__
