#ifndef ist_GraphicsCommon_Vertex_h
#define ist_GraphicsCommon_Vertex_h

namespace ist {

enum VertexType {
    VT_Unknown,
    VT_P2C4,
    VT_P2T2C4,
    VT_P3T2C4,
    VT_End,
};

#define istEachVertexTypes(F)\
    F(VertexP2C4)\
    F(VertexP2T2C4)\
    F(VertexP3T2C4)


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

template<class T> struct GetVertexTypeID { static const VertexType result=VT_Unknown; };
template<> struct GetVertexTypeID<VertexP2C4>     { static const VertexType result=VT_P2C4; };
template<> struct GetVertexTypeID<VertexP2T2C4>   { static const VertexType result=VT_P2T2C4; };
template<> struct GetVertexTypeID<VertexP3T2C4>   { static const VertexType result=VT_P3T2C4; };


} // namespace ist
#endif // ist_GraphicsCommon_Vertex_h
