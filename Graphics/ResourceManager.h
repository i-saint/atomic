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
    MODEL_QUAD_SCREEN,
    MODEL_QUAD_VFX,
    MODEL_CUBE_FRACTION,
    MODEL_CUBE_VFX,
    MODEL_OCTAHEDRON_BULLET,
    MODEL_SPHERE_LIGHT,

    MODEL_END,
};
enum SH_RID {
    SH_GBUFFER,
    SH_GBUFFER_OCTAHEDRON,
    SH_DEFERRED,
    SH_BLOOM,
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

enum VBO_RID {
    VBO_BOX_POS,
    VBO_CUBE_POS,
    VBO_CUBE_SCALE,
    VBO_CUBE_GLOW,
    VBO_OCTAHEDRON_POS,
    VBO_OCTAHEDRON_SCALE,
    VBO_OCTAHEDRON_DIR,
    VBO_OCTAHEDRON_SEED,
    VBO_OCTAHEDRON_TIME,
    VBO_SPHERE_LIGHT_POS,
    VBO_SPHERE_LIGHT_SCALE,
    VBO_END,
};

enum UBO_RID {
    UBO_DUMMY,
    UBO_END,
};

enum CLP_RID {
    CLP_FRACTION_UPDATE,
    CLP_END,
};

enum CLB_RID {
    CLB_FRACTION,
    CLB_END,
};


typedef Color3DepthBuffer RenderTargetGBuffer;
typedef ColorDepthBuffer RenderTargetDeferred;
typedef ColorBuffer RenderTargetGauss;

class GraphicResourceManager : boost::noncopyable
{
private:

    ShaderGBuffer               *m_sh_gbuffer;
    ShaderGBuffer_Octahedron    *m_sh_gbuffer_octahedron;
    ShaderDeferred              *m_sh_deferred;
    ShaderBloom                 *m_sh_bloom;
    ShaderOutput                *m_sh_output;

    RenderTargetGBuffer     *m_rt_gbuffer;
    RenderTargetDeferred    *m_rt_deferred;
    ColorBuffer             *m_rt_gauss[2];

    ModelData           *m_model[MODEL_END];
    Texture2D           *m_tex2d[TEX2D_END];
    VertexBufferObject  *m_vbo[VBO_END];
    UniformBufferObject *m_ubo[UBO_END];
    FrameBufferObject   *m_fbo[RT_END];
    ProgramObject       *m_shader[SH_END];

    cl::Program         *m_cl_programs[CLP_END];
    cl::Buffer          *m_cl_buffers[CLB_END];

private:
    static GraphicResourceManager* s_inst;
    bool initialize();
    void finalize();

public:
    static GraphicResourceManager* getInstance() { return s_inst; }
    static void intializeInstance();
    static void finalizeInstance();

    ModelData* getModelData(MODEL_RID i)                    { return m_model[i]; }
    Texture2D* getTexture2D(TEX2D_RID i)                    { return m_tex2d[i]; }
    VertexBufferObject* getVertexBufferObject(VBO_RID i)    { return m_vbo[i]; }
    UniformBufferObject* getUniformBufferObject(UBO_RID i)  { return m_ubo[i]; }
    cl::Program* getCLProgram(CLP_RID i)                    { return m_cl_programs[i]; }
    cl::Buffer* getCLBuffer(CLB_RID i)                      { return m_cl_buffers[i]; }

    ShaderGBuffer*              getShaderGBuffer()              { return m_sh_gbuffer; }
    ShaderGBuffer_Octahedron*   getShaderGBuffer_Octahedron()   { return m_sh_gbuffer_octahedron; }
    ShaderDeferred*             getShaderDeferred()             { return m_sh_deferred; }
    ShaderBloom*                getShaderBloom()                { return m_sh_bloom; }
    ShaderOutput*               getShaderOutput()               { return m_sh_output; }

    void swapBuffers();
    RenderTargetGBuffer*    getRenderTargetGBuffer()        { return m_rt_gbuffer; }
    RenderTargetDeferred*   getRenderTargetDeferred()       { return m_rt_deferred; }
    RenderTargetGauss*      getRenderTargetGauss(uint32 i)  { return m_rt_gauss[i]; }
};


#define atomicGetResourceManager()   GraphicResourceManager::getInstance()

#define atomicGetRenderTargetGBuffer()      atomicGetResourceManager()->getRenderTargetGBuffer()
#define atomicGetRenderTargetDeferred()     atomicGetResourceManager()->getRenderTargetDeferred()
#define atomicGetRenderTargetGauss(i)       atomicGetResourceManager()->getRenderTargetGauss(i)

#define atomicGetShaderGBuffer()            atomicGetResourceManager()->getShaderGBuffer()
#define atomicGetShaderGBuffer_Octahedron() atomicGetResourceManager()->getShaderGBuffer_Octahedron()
#define atomicGetShaderDeferred()           atomicGetResourceManager()->getShaderDeferred()
#define atomicGetShaderBloom()              atomicGetResourceManager()->getShaderBloom()
#define atomicGetShaderOutput()             atomicGetResourceManager()->getShaderOutput()

#define atomicGetModelData(i)               atomicGetResourceManager()->getModelData(i)
#define atomicGetTexture2D(i)               atomicGetResourceManager()->getTexture2D(i)
#define atomicGetVertexBufferObject(i)      atomicGetResourceManager()->getVertexBufferObject(i)
#define atomicGetUniformBufferObject(i)     atomicGetResourceManager()->getUniformBufferObject(i)

#define atomicGetCLProgram(i)               atomicGetResourceManager()->getCLProgram(i)
#define atomicGetCLBuffer(i)                atomicGetResourceManager()->getCLBuffer(i)

} // namespace atomic
#endif // __atomic_Graphics_ResourceManager_h__
