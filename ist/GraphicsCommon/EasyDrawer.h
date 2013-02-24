#ifndef ist_Graphics_EasyDrawer_h
#define ist_Graphics_EasyDrawer_h

#include "ist/Base.h"
#include "ist/Graphics.h"

#define ist_EasyDraw_impl_GL


namespace ist {
#ifdef ist_EasyDraw_impl_GL
namespace i3dgl {
#endif // ist_EasyDraw_impl_GL


struct VertexP2C4
{
    vec2 position;
    vec4 color;

    VertexP2C4(const vec2 &p=vec2(), const vec4 &c=vec4())
        : position(p), color(c)
    {}
};
struct VertexP2T2C4
{
    vec2 position;
    vec2 texcoord;
    vec4 color;

    VertexP2T2C4(const vec2 &p=vec2(), const vec2 &t=vec2(), const vec4 &c=vec4())
        : position(p), texcoord(t), color(c)
    {}
};
struct VertexP3T2C4
{
    vec3 position;
    vec2 texcoord;
    vec4 color;

    VertexP3T2C4(const vec3 &p=vec3(), const vec2 &t=vec2(), const vec4 &c=vec4())
        : position(p), texcoord(t), color(c)
    {}
};

enum VertexType {
    VT_Unknown,
    VT_P2C4,
    VT_P2T2C4,
    VT_P3T2C4,
    VT_End,
};
template<class T> struct GetVertexType { static const VertexType result=VT_Unknown; };
template<> struct GetVertexType<VertexP2C4>     { static const VertexType result=VT_P2C4; };
template<> struct GetVertexType<VertexP2T2C4>   { static const VertexType result=VT_P2T2C4; };
template<> struct GetVertexType<VertexP3T2C4>   { static const VertexType result=VT_P3T2C4; };

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
    mat4 m_proj;
    mat4 m_world;
    Texture2D *m_texture;
    Sampler *m_sampler;
    ShaderProgram *m_shader;
    uint32 m_uniform_location;
};


class istInterModule EasyDrawer : public boost::noncopyable
{
friend EasyDrawer* CreateEasyDrawer(Device *dev);
public:
    struct DrawCall
    {
        EasyDrawState state;
        I3D_TOPOLOGY topology;
        VertexType vertex_type;
        uint32 num_vertices;
        uint32 buffer_index; // in byte
    };

public:
    void release();
    void flush(DeviceContext *ctx);

    template<class VertexT>
    void draw(const EasyDrawState &state, I3D_TOPOLOGY topology, const VertexT *vertices, uint32 num_vertices);

private:
    EasyDrawer(Device *dev);
    ~EasyDrawer();
    void updateBuffers(DeviceContext *ctx);

    struct Members;
    deep_copy_ptr<Members> m;
};

istInterModule EasyDrawer* CreateEasyDrawer(Device *dev);



istInterModule void DrawLine(
    EasyDrawer &drawer, const EasyDrawState &state,
    const vec2 &pos1, const vec2 &pos2,
    const vec4 &color);
istInterModule void DrawLine(
    EasyDrawer &drawer, const EasyDrawState &state,
    const vec2 &pos1, const vec2 &pos2,
    const vec4 &color1, const vec4 &color2);

istInterModule void DrawOutlineRect(
    EasyDrawer &drawer, const EasyDrawState &state,
    const vec2 &ur, const vec4 &bl,
    const vec4 &color );
istInterModule void DrawOutlineRect(
    EasyDrawer &drawer, const EasyDrawState &state,
    const vec2 &ur, const vec2 &bl,
    const vec4 &cur, const vec4 &cul, const vec4 &cbl, const vec4 &cbr );

istInterModule void DrawRect(
    EasyDrawer &drawer, const EasyDrawState &state,
    const vec2 &ur, const vec2 &bl,
    const vec4 &color );
istInterModule void DrawRect(
    EasyDrawer &drawer, const EasyDrawState &state,
    const vec2 &ur, const vec2 &bl,
    const vec4 &cur, const vec4 &cul, const vec4 &cbl, const vec4 &cbr );
istInterModule void DrawRectT(
    EasyDrawer &drawer, const EasyDrawState &state,
    const vec2 &ur, const vec2 &bl,
    const vec2 &tur=vec2(1.0f,1.0f), const vec2 &tbl=vec2(0.0f,0.0f),
    const vec4 &color=vec4(1.0f) );




} // namespace i3d*
} // namespace ist

#endif // ist_Graphics_EasyDrawer_h
