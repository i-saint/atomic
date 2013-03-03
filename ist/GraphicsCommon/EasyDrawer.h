#ifndef ist_GraphicsCommon_EasyDrawer_h
#define ist_GraphicsCommon_EasyDrawer_h

#include "ist/Base.h"
#include "ist/Graphics.h"
#include "Vertex.h"

#define ist_EasyDrawer_impl_GL

#ifdef ist_EasyDrawer_impl_GL
#   define ist_EasyDrawer_NamespaceBegin  namespace ist{ namespace i3dgl{
#   define ist_EasyDraw_NamespaceEnd    }}
#else
// todo
#endif


ist_EasyDrawer_NamespaceBegin

extern const char *g_vs_p2c4;
extern const char *g_vs_p2t2c4;
extern const char *g_vs_p3t2c4;

class istInterModule EasyDrawState
{
public:
    EasyDrawState();
    void setScreen(float32 width, float32 height);
    void setScreen(float32 left, float32 right, float32 bottom, float32 top);
    void setProjectionMatrix(const mat4 &mat);
    void setWorldMatrix(const mat4 &mat);
    void setTexture(Texture2D *tex);
    void setSampler(Sampler *smp);
    void setShader(ShaderProgram *smp);

    const mat4&     getProjectionMatrix() const;
    const mat4&     getWorldMatrix() const;
    Texture2D*      getTexture() const;
    Sampler*        getSampler() const;
    ShaderProgram*  getShader() const;
    uint32          getUniformLocation() const;

private:
    istMemberPtrDecl(Members) m;
};


class istInterModule EasyDrawer : public SharedObject
{
friend istInterModule EasyDrawer* CreateEasyDrawer();
public:
    const EasyDrawState& getRenderStates();
    void forceSetRenderStates(const EasyDrawState &ds);

    void setScreen(float32 width, float32 height);
    void setScreen(float32 left, float32 right, float32 bottom, float32 top);
    void setProjectionMatrix(const mat4 &mat);
    void setWorldMatrix(const mat4 &mat);
    void setTexture(Texture2D *tex);
    void setSampler(Sampler *smp);
    void setShader(ShaderProgram *smp);

    const mat4&     getProjectionMatrix() const;
    const mat4&     getWorldMatrix() const;
    Texture2D*      getTexture() const;
    Sampler*        getSampler() const;
    ShaderProgram*  getShader() const;
    uint32          getUniformLocation() const;

    // template は dllexport できないのでマクロによる展開で代用
#define Template(VertexT)\
    void draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices);\
    void draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint8 *indices, uint32 num_indices);\
    void draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint16 *indices, uint32 num_indices);\
    void draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const uint32 *indices, uint32 num_indices);
    istEachVertexTypes(Template)
#undef Template

    void flush(DeviceContext *ctx);

private:
    EasyDrawer();
    ~EasyDrawer();
    void updateBuffers(DeviceContext *ctx);

    template<class VertexT>
    void drawImpl(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices);
    template<class VertexT, class IndexT>
    void drawImpl(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices, const IndexT *indices, uint32 num_indices);

    struct DrawCall;
    istMemberPtrDecl(Members) m;
};

istInterModule EasyDrawer* CreateEasyDrawer();

ist_EasyDraw_NamespaceEnd


#include "EasyDrawerUtil.h"
#endif // ist_GraphicsCommon_EasyDrawer_h
