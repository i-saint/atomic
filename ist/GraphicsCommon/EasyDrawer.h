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

class istAPI EasyDrawState
{
public:
    EasyDrawState();
    void setViewport(int32 width, int32 height);
    void setViewport(int32 x, int32 y, int32 width, int32 height);
    void setScreen(float32 width, float32 height);
    void setScreen(float32 left, float32 right, float32 bottom, float32 top);
    void setProjectionMatrix(const mat4 &mat);
    void setWorldMatrix(const mat4 &mat);
    void setTexture(Texture2D *tex);
    void setSampler(Sampler *smp);
    void setShader(ShaderProgram *smp);

    const Viewport& getViewport() const;
    const mat4&     getProjectionMatrix() const;
    const mat4&     getWorldMatrix() const;
    Texture2D*      getTexture() const;
    Sampler*        getSampler() const;
    ShaderProgram*  getShader() const;
    uint32          getUniformLocation() const;

private:
    mat4            m_proj;
    mat4            m_world;
    Texture2D      *m_texture;
    Sampler        *m_sampler;
    ShaderProgram  *m_shader;
    Viewport        m_viewport;
    uint32          m_uniform_location;
};


class istAPI EasyDrawer : public SharedObject
{
friend istAPI EasyDrawer* CreateEasyDrawer();
public:
    const EasyDrawState& getRenderStates();
    void forceSetRenderStates(const EasyDrawState &ds);

    void setViewport(int32 width, int32 height);
    void setViewport(int32 x, int32 y, int32 width, int32 height);
    void setScreen(float32 width, float32 height);
    void setScreen(float32 left, float32 right, float32 bottom, float32 top);
    void setProjectionMatrix(const mat4 &mat);
    void setWorldMatrix(const mat4 &mat);
    void setTexture(Texture2D *tex);
    void setSampler(Sampler *smp);
    void setShader(ShaderProgram *smp);

    const Viewport& getViewport() const;
    const mat4&     getProjectionMatrix() const;
    const mat4&     getWorldMatrix() const;
    Texture2D*      getTexture() const;
    Sampler*        getSampler() const;
    ShaderProgram*  getShader() const;
    uint32          getUniformLocation() const;

    // template は dllexport できないのでマクロによる展開で代用
#define Template(VertexT)\
    void draw(I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices);\
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

    class EasyShaders;
    struct DrawCall
    {
        EasyDrawState state;
        I3D_TOPOLOGY topology;
        VertexType vertex_type;
        I3D_TYPE index_type;
        uint32 num_vertices;
        size_t vb_offset; // in byte
        uint32 num_indices;
        size_t ib_offset; // in byte
    };
    VertexArray            *m_va;
    Buffer                 *m_vbo;
    Buffer                 *m_ibo;
    Buffer                 *m_ubo;
    EasyShaders            *m_shaders;
    ist::raw_vector<char>   m_vertex_data;
    ist::raw_vector<char>   m_index_data;
    ist::vector<DrawCall>   m_draw_calls;
    EasyDrawState           m_state;
};

istAPI EasyDrawer* CreateEasyDrawer();

ist_EasyDraw_NamespaceEnd


#include "EasyDrawerUtil.h"
#endif // ist_GraphicsCommon_EasyDrawer_h
