#ifndef __atomic_Graphics_ResourceManager__
#define __atomic_Graphics_ResourceManager__

#include "Graphics/Shader.h"
#include "Graphics/CudaBuffer.h"
#include "Graphics/ParticleSet.h"
#include "Graphics/ResourceID.h"
#include "Graphics/CreateModelData.h"
#include "GPGPU/SPH.cuh"

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
    GBUFFER_NORMAL  = 1,
    GBUFFER_POSITION= 2,
    GBUFFER_GLOW    = 3,
};


class GraphicResourceManager : boost::noncopyable
{
private:
    IFontRenderer   *m_font;
    Sampler         *m_sampler[SAMPLER_END];
    Texture1D       *m_tex1d[TEX1D_END];
    Texture2D       *m_tex2d[TEX2D_END];
    VertexArray     *m_va[VA_END];
    Buffer          *m_vbo[VBO_END];
    Buffer          *m_ibo[IBO_END];
    Buffer          *m_ubo[UBO_END];
    RenderTarget    *m_rt[RT_END];
    AtomicShader    *m_shader[SH_END];
    ParticleSet     m_pset[PSET_END];
    RigidInfo       m_rinfo[PSET_END];

private:
    static GraphicResourceManager* s_inst;
    bool initialize();
    void finalize();

public:
    static GraphicResourceManager* getInstance() { return s_inst; }
    static void intializeInstance();
    static void finalizeInstance();

    IFontRenderer*  getFont()                   { return m_font; }
    Sampler*        getSampler(SAMPLER_RID i)   { return m_sampler[i]; }
    Texture1D*      getTexture1D(TEX1D_RID i)   { return m_tex1d[i]; }
    Texture2D*      getTexture2D(TEX2D_RID i)   { return m_tex2d[i]; }
    VertexArray*    getVertexArray(VA_RID i)    { return m_va[i]; }
    Buffer*         getVertexBuffer(VBO_RID i)  { return m_vbo[i]; }
    Buffer*         getIndexBuffer(IBO_RID i)   { return m_ibo[i]; }
    Buffer*         getUniformBuffer(UBO_RID i) { return m_ubo[i]; }
    AtomicShader*   getShader(SH_RID i)         { return m_shader[i]; }
    RenderTarget*   getRenderTarget(RT_RID i)   { return m_rt[i]; }
    ParticleSet*    getParticleSet(PSET_RID i)  { return &m_pset[i]; }
    RigidInfo*      getRigidInfo(PSET_RID i)    { return &m_rinfo[i]; }
};


#define atomicGetResourceManager()   GraphicResourceManager::getInstance()

#define atomicGetFont()             atomicGetResourceManager()->getFont()
#define atomicGetSampler(i)         atomicGetResourceManager()->getSampler(i)
#define atomicGetTexture1D(i)       atomicGetResourceManager()->getTexture1D(i)
#define atomicGetTexture2D(i)       atomicGetResourceManager()->getTexture2D(i)
#define atomicGetVertexArray(i)     atomicGetResourceManager()->getVertexArray(i)
#define atomicGetVertexBuffer(i)    atomicGetResourceManager()->getVertexBuffer(i)
#define atomicGetIndexBuffer(i)     atomicGetResourceManager()->getIndexBuffer(i)
#define atomicGetUniformBuffer(i)   atomicGetResourceManager()->getUniformBuffer(i)
#define atomicGetShader(i)          atomicGetResourceManager()->getShader(i)
#define atomicGetRenderTarget(i)    atomicGetResourceManager()->getRenderTarget(i)
#define atomicGetParticleSet(i)     atomicGetResourceManager()->getParticleSet(i)
#define atomicGetRigidInfo(i)       atomicGetResourceManager()->getRigidInfo(i)

} // namespace atomic
#endif // __atomic_Graphics_ResourceManager__
