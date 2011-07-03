#ifndef __atomic_Graphics_GraphicResourceManager__
#define __atomic_Graphics_GraphicResourceManager__
#include "Shader.h"

namespace atomic {

void DrawScreen(vec2 min_pos, vec2 max_pos, vec2 min_tc, vec2 max_tc);
void DrawScreen(vec2 min_tc, vec2 max_tc);
void DrawScreen();


enum PASS {
    PASS_SHADOW_DEPTH,
    PASS_GBUFFER,
    PASS_DEFERRED,
    PASS_FORWARD,
    PASS_POSTPROCESS,
    PASS_UI,
    PASS_END,
};

enum MODEL_INDEX {
    MODEL_QUAD,
    MODEL_CUBE,
    MODEL_SPHERE,
    MODEL_END,
};
enum SH_INDEX {
    SH_GBUFFER,
    SH_DEFERRED,
    SH_BLOOM,
    SH_OUTPUT,
    SH_END,
};
enum RT_INDEX {
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

enum TEX2D_INDEX {
    TEX2D_RANDOM,
    TEX2D_END,
};

enum VBO_INDEX {
    VBO_BOX_POS,
    VBO_CUBE_POS,
    VBO_CUBE_SCALE,
    VBO_CUBE_GLOW,
    VBO_SPHERE_LIGHT_POS,
    VBO_SPHERE_LIGHT_SCALE,
    VBO_END,
};

enum UBO_INDEX {
    UBO_DUMMY,
    UBO_END,
};

typedef Color3DepthBuffer RenderTargetGBuffer;
typedef ColorDepthBuffer RenderTargetDeferred;
typedef ColorBuffer RenderTargetGauss;

class GraphicResourceManager : boost::noncopyable
{
private:

    ShaderGBuffer   *m_sh_gbuffer;
    ShaderDeferred  *m_sh_deferred;
    ShaderBloom     *m_sh_bloom;
    ShaderOutput    *m_sh_output;

    RenderTargetGBuffer     *m_rt_gbuffer;
    RenderTargetDeferred    *m_rt_deferred;
    ColorBuffer             *m_rt_gauss[2];

    ModelData           *m_model[MODEL_END];
    Texture2D           *m_tex2d[TEX2D_END];
    VertexBufferObject  *m_vbo[VBO_END];
    UniformBufferObject *m_ubo[UBO_END];
    FrameBufferObject   *m_fbo[RT_END];
    ProgramObject       *m_shader[SH_END];

private:
    static GraphicResourceManager* s_inst;
    bool initialize();
    void finalize();

public:
    static GraphicResourceManager* getInstance() { return s_inst; }
    static void intializeInstance();
    static void finalizeInstance();

    ModelData* getModelData(MODEL_INDEX i)                      { return m_model[i]; }
    Texture2D* getTexture2D(TEX2D_INDEX i)                      { return m_tex2d[i]; }
    VertexBufferObject* getVertexBufferObject(VBO_INDEX i)      { return m_vbo[i]; }
    UniformBufferObject* getUniformBufferObject(UBO_INDEX i)    { return m_ubo[i]; }

    ShaderGBuffer*  getShaderGBuffer()      { return m_sh_gbuffer; }
    ShaderDeferred* getShaderDeferred()     { return m_sh_deferred; }
    ShaderBloom*    getShaderBloom()        { return m_sh_bloom; }
    ShaderOutput*   getShaderOutput()       { return m_sh_output; }

    void swapBuffers();
    RenderTargetGBuffer*    getRenderTargetGBuffer()        { return m_rt_gbuffer; }
    RenderTargetDeferred*   getRenderTargetDeferred()       { return m_rt_deferred; }
    RenderTargetGauss*      getRenderTargetGauss(uint32 i)  { return m_rt_gauss[i]; }
};


#define GetGraphicResourceManager() GraphicResourceManager::getInstance()

#define GetRenderTargetGBuffer()            GetGraphicResourceManager()->getRenderTargetGBuffer()
#define GetRenderTargetDeferred()           GetGraphicResourceManager()->getRenderTargetDeferred()
#define GetRenderTargetGauss(i)             GetGraphicResourceManager()->getRenderTargetGauss(i)

#define GetShaderGBuffer()                  GetGraphicResourceManager()->getShaderGBuffer()
#define GetShaderDeferred()                 GetGraphicResourceManager()->getShaderDeferred()
#define GetShaderBloom()                    GetGraphicResourceManager()->getShaderBloom()
#define GetShaderOutput()                   GetGraphicResourceManager()->getShaderOutput()

#define GetModelData(i)             GetGraphicResourceManager()->getModelData(i)
#define GetTexture2D(i)             GetGraphicResourceManager()->getTexture2D(i)
#define GetVertexBufferObject(i)    GetGraphicResourceManager()->getVertexBufferObject(i)
#define GetUniformBufferObject(i)   GetGraphicResourceManager()->getUniformBufferObject(i)

} // namespace atomic
#endif // __atomic_Graphics_GraphicResourceManager__
