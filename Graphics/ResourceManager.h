#ifndef __atomic_Graphics_ResourceManager_h__
#define __atomic_Graphics_ResourceManager_h__
#include "Graphics/Shader.h"
#include "Graphics/CreateModelData.h"

namespace atomic {

void DrawScreen(vec2 min_pos, vec2 max_pos, vec2 min_tc, vec2 max_tc);
void DrawScreen(vec2 min_tc, vec2 max_tc);
void DrawScreen();


enum DRAW_PASS {
    PASS_SHADOW_DEPTH,
    PASS_GBUFFER,
    PASS_DEFERRED,
    PASS_FORWARD,
    PASS_POSTPROCESS,
    PASS_UI,
    PASS_END,
};

enum MODEL_RID {
    MODEL_SPHERE_LIGHT,
    MODEL_END,
};
enum SH_RID {
    SH_GBUFFER,
    SH_POINTLIGHT,
    SH_BLOOM_LUMINANCE,
    SH_BLOOM_HBLUR,
    SH_BLOOM_VBLUR,
    SH_BLOOM_COMPOSITE,
    SH_NORMAL_HBLUR,
    SH_NORMAL_VBLUR,
    SH_OUTPUT,
    SH_END,
};
enum RT_RID {
    RT_GBUFFER,
    RT_DEFERRED,
    RT_GAUSS0,
    RT_GAUSS1,
    RT_END,
};

enum GBUFFER {
    GBUFFER_COLOR       = Texture2D::SLOT_0,
    GBUFFER_GLOW        = Texture2D::SLOT_1,
    GBUFFER_NORMAL      = Texture2D::SLOT_2,
    GBUFFER_POSITION    = Texture2D::SLOT_3,
    GBUFFER_DEPTH       = Texture2D::SLOT_4,
};

enum TEX2D_RID {
    TEX2D_RANDOM,
    TEX2D_END,
};

enum VA_RID {
    VA_SCREEN_QUAD,
    VA_BLOOM_LUMINANCE_QUADS,
    VA_BLOOM_BLUR_QUADS,
    VA_BLOOM_COMPOSITE_QUAD,
    VA_UNIT_CUBE,
    VA_UNIT_SPHERE,
    VA_FRACTION_CUBE,
    VA_END,
};

enum VBO_RID {
    VBO_SCREEN_QUAD,
    VBO_BLOOM_LUMINANCE_QUADS,
    VBO_BLOOM_BLUR_QUADS,
    VBO_BLOOM_COMPOSITE_QUAD,
    VBO_UNIT_CUBE,
    VBO_UNIT_SPHERE,
    VBO_FRACTION_CUBE,

    VBO_FRACTION_INSTANCE,
    VBO_RIGID_FRACTION_POS,
    VBO_POINTLIGHT_INSTANCE,
    VBO_POINTLIGHT_SCALE,
    VBO_END,
};

enum IBO_RID {
    IBO_SPHERE,
    IBO_END,
};

enum UBO_RID {
    UBO_RENDER_STATES,
    UBO_BLOOM_STATES,
    UBO_END,
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
    ModelData           *m_model[MODEL_END];
    Texture2D           *m_tex2d[TEX2D_END];
    VertexArray         *m_va[VA_END];
    VertexBufferObject  *m_vbo[VBO_END];
    IndexBufferObject   *m_ibo[IBO_END];
    UniformBufferObject *m_ubo[UBO_END];
    FrameBufferObject   *m_fbo[RT_END];
    AtomicShader        *m_shader[SH_END];

private:
    static GraphicResourceManager* s_inst;
    bool initialize();
    void finalize();

public:
    static GraphicResourceManager* getInstance() { return s_inst; }
    static void intializeInstance();
    static void finalizeInstance();

    SystemFont* getFont() { return m_font; }
    ModelData* getModelData(MODEL_RID i)                    { return m_model[i]; }
    Texture2D* getTexture2D(TEX2D_RID i)                    { return m_tex2d[i]; }
    VertexArray* getVertexArray(VA_RID i)                   { return m_va[i]; }
    VertexBufferObject* getVertexBufferObject(VBO_RID i)    { return m_vbo[i]; }
    IndexBufferObject* getIndexBufferObject(IBO_RID i)      { return m_ibo[i]; }
    UniformBufferObject* getUniformBufferObject(UBO_RID i)  { return m_ubo[i]; }
    AtomicShader* getShader(SH_RID i)                       { return m_shader[i]; }

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
#define atomicGetModelData(i)               atomicGetResourceManager()->getModelData(i)
#define atomicGetTexture2D(i)               atomicGetResourceManager()->getTexture2D(i)
#define atomicGetVertexArray(i)             atomicGetResourceManager()->getVertexArray(i)
#define atomicGetVertexBufferObject(i)      atomicGetResourceManager()->getVertexBufferObject(i)
#define atomicGetIndexBufferObject(i)       atomicGetResourceManager()->getIndexBufferObject(i)
#define atomicGetUniformBufferObject(i)     atomicGetResourceManager()->getUniformBufferObject(i)
#define atomicGetShader(i)                  atomicGetResourceManager()->getShader(i)

} // namespace atomic
#endif // __atomic_Graphics_ResourceManager_h__
