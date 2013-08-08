#ifndef atm_Graphics_ResourceManager_h
#define atm_Graphics_ResourceManager_h

#include "Graphics/Shader.h"
#include "Graphics/ParticleSet.h"
#include "Graphics/ResourceID.h"
#include "Graphics/CreateModelData.h"

namespace atm {

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


class atmAPI istAlign(16) GraphicResourceManager
{
istNonCopyable(GraphicResourceManager);
istMakeDestructable;
public:
    static GraphicResourceManager* getInstance() { return s_inst; }
    static void intializeInstance();
    static void finalizeInstance();

    void update();

    IFontRenderer*      getFont()                   { return m_font; }
    IFontRenderer*      getTitleFont()              { return m_title_font; }
    Sampler*            getSampler(SAMPLER_RID i)   { return m_sampler[i]; }
    Texture1D*          getTexture1D(TEX1D_RID i)   { return m_tex1d[i]; }
    Texture2D*          getTexture2D(TEX2D_RID i)   { return m_tex2d[i]; }
    VertexArray*        getVertexArray(VA_RID i)    { return m_va[i]; }
    Buffer*             getVertexBuffer(VBO_RID i)  { return m_vbo[i]; }
    Buffer*             getIndexBuffer(IBO_RID i)   { return m_ibo[i]; }
    Buffer*             getUniformBuffer(UBO_RID i) { return m_ubo[i]; }
    AtomicShader*       getShader(SH_RID i)         { return m_shader[i]; }
    RenderTarget*       getRenderTarget(RT_RID i)   { return m_rt[i]; }
    ParticleSet*        getParticleSet(PSET_RID i)  { return &m_pset[i]; }
    RigidInfo*          getRigidInfo(PSET_RID i)    { return &m_rinfo[i]; }
    ModelInfo*          getModelInfo(MODEL_RID i)   { return &m_models[i]; }
    BlendState*         getBlendState(BLEND_RID i)          { return m_blend_states[i]; }
    DepthStencilState*  getDepthStencilState(DEPTH_RID i)   { return m_depth_states[i]; }

private:
    static GraphicResourceManager* s_inst;
    bool initialize();
    void finalize();

private:
    GraphicResourceManager();
    ~GraphicResourceManager();

    ParticleSet         m_pset[PSET_END];
    RigidInfo           m_rinfo[PSET_END];
    ModelInfo           m_models[MODEL_END];
    IFontRenderer       *m_font;
    IFontRenderer       *m_title_font;
    Sampler             *m_sampler[SAMPLER_END];
    Texture1D           *m_tex1d[TEX1D_END];
    Texture2D           *m_tex2d[TEX2D_END];
    VertexArray         *m_va[VA_END];
    Buffer              *m_vbo[VBO_END];
    Buffer              *m_ibo[IBO_END];
    Buffer              *m_ubo[UBO_END];
    RenderTarget        *m_rt[RT_END];
    AtomicShader        *m_shader[SH_END];
    BlendState          *m_blend_states[BS_END];
    DepthStencilState   *m_depth_states[DS_END];
    bool m_flag_exit;

#ifdef atm_enable_ShaderLiveEdit
    void watchGLSLFiles();
    bool                m_glsl_modified;
    HANDLE              m_glsl_notifier;
    ist::FunctorThread<std::function<void ()> > m_glsl_watcher;
#endif // atm_enable_ShaderLiveEdit
};


#define atmGetGraphicsResourceManager()   GraphicResourceManager::getInstance()

#define atmGetFont()             atmGetGraphicsResourceManager()->getFont()
#define atmGetTitleFont()        atmGetGraphicsResourceManager()->getTitleFont()
#define atmGetSampler(i)         atmGetGraphicsResourceManager()->getSampler(i)
#define atmGetTexture1D(i)       atmGetGraphicsResourceManager()->getTexture1D(i)
#define atmGetTexture2D(i)       atmGetGraphicsResourceManager()->getTexture2D(i)
#define atmGetVertexArray(i)     atmGetGraphicsResourceManager()->getVertexArray(i)
#define atmGetVertexBuffer(i)    atmGetGraphicsResourceManager()->getVertexBuffer(i)
#define atmGetIndexBuffer(i)     atmGetGraphicsResourceManager()->getIndexBuffer(i)
#define atmGetUniformBuffer(i)   atmGetGraphicsResourceManager()->getUniformBuffer(i)
#define atmGetShader(i)          atmGetGraphicsResourceManager()->getShader(i)
#define atmGetRenderTarget(i)    atmGetGraphicsResourceManager()->getRenderTarget(i)
#define atmGetParticleSet(i)     atmGetGraphicsResourceManager()->getParticleSet(i)
#define atmGetRigidInfo(i)       atmGetGraphicsResourceManager()->getRigidInfo(i)
#define atmGetModelInfo(i)       atmGetGraphicsResourceManager()->getModelInfo(i)
#define atmGetBlendState(i)          atmGetGraphicsResourceManager()->getBlendState(i)
#define atmGetDepthStencilState(i)   atmGetGraphicsResourceManager()->getDepthStencilState(i)

} // namespace atm
#endif // atm_Graphics_ResourceManager_h
