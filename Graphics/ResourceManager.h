#ifndef __atomic_Graphics_ResourceManager__
#define __atomic_Graphics_ResourceManager__
#include "Graphics/Shader.h"
#include "Graphics/CreateModelData.h"
#include "Graphics/CudaBuffer.h"
#include "Graphics/ResourceID.h"

namespace atomic {

enum DRAW_PASS {
    PASS_SHADOW_DEPTH,
    PASS_GBUFFER,
    PASS_DEFERRED,
    PASS_FORWARD,
    PASS_POSTPROCESS,
    PASS_HUD,
    PASS_END,
};


enum GBUFFER {
    GBUFFER_COLOR   = 0,
    GBUFFER_GLOW    = 1,
    GBUFFER_NORMAL  = 2,
    GBUFFER_POSITION= 3,
    GBUFFER_DEPTH   = 4,
};


typedef Color3DepthBuffer RenderTargetGBuffer;
typedef ColorDepthBuffer RenderTargetDeferred;
typedef ColorBuffer RenderTargetGauss;

class GraphicResourceManager : boost::noncopyable
{
private:
    RenderTargetGBuffer     *m_rt_gbuffer;
    RenderTargetDeferred    *m_rt_deferred;
    ColorBuffer             *m_rt_gauss[2];

    SystemFont          *m_font;
    Texture2D           *m_tex2d[TEX2D_END];
    VertexArray         *m_va[VA_END];
    VertexBufferObject  *m_vbo[VBO_END];
    IndexBufferObject   *m_ibo[IBO_END];
    UniformBufferObject *m_ubo[UBO_END];
    FrameBufferObject   *m_fbo[RT_END];
    AtomicShader        *m_shader[SH_END];
    CudaBuffer          *m_cb[CB_END];

private:
    static GraphicResourceManager* s_inst;
    bool initialize();
    void finalize();

public:
    static GraphicResourceManager* getInstance() { return s_inst; }
    static void intializeInstance();
    static void finalizeInstance();

    SystemFont* getFont() { return m_font; }
    Texture2D* getTexture2D(TEX2D_RID i)                    { return m_tex2d[i]; }
    VertexArray* getVertexArray(VA_RID i)                   { return m_va[i]; }
    VertexBufferObject* getVertexBufferObject(VBO_RID i)    { return m_vbo[i]; }
    IndexBufferObject* getIndexBufferObject(IBO_RID i)      { return m_ibo[i]; }
    UniformBufferObject* getUniformBufferObject(UBO_RID i)  { return m_ubo[i]; }
    AtomicShader* getShader(SH_RID i)                       { return m_shader[i]; }
    CudaBuffer* getCudaBuffer(CB_RID i)                     { return m_cb[i]; }

    void swapBuffers();
    RenderTargetGBuffer*    getRenderTargetGBuffer()        { return m_rt_gbuffer; }
    RenderTargetDeferred*   getRenderTargetDeferred()       { return m_rt_deferred; }
    RenderTargetGauss*      getRenderTargetGauss(uint32 i)  { return m_rt_gauss[i]; }
};


#define atomicGetResourceManager()   GraphicResourceManager::getInstance()

#define atomicGetRenderTargetGBuffer()      atomicGetResourceManager()->getRenderTargetGBuffer()
#define atomicGetRenderTargetDeferred()     atomicGetResourceManager()->getRenderTargetDeferred()
#define atomicGetRenderTargetGauss(i)       atomicGetResourceManager()->getRenderTargetGauss(i)

#define atomicGetFont()                     atomicGetResourceManager()->getFont()
#define atomicGetTexture2D(i)               atomicGetResourceManager()->getTexture2D(i)
#define atomicGetVertexArray(i)             atomicGetResourceManager()->getVertexArray(i)
#define atomicGetVertexBufferObject(i)      atomicGetResourceManager()->getVertexBufferObject(i)
#define atomicGetIndexBufferObject(i)       atomicGetResourceManager()->getIndexBufferObject(i)
#define atomicGetUniformBufferObject(i)     atomicGetResourceManager()->getUniformBufferObject(i)
#define atomicGetShader(i)                  atomicGetResourceManager()->getShader(i)
#define atomicGetParticleSet(i)             atomicGetResourceManager()->getCudaBuffer(i)

} // namespace atomic
#endif // __atomic_Graphics_ResourceManager__
