#ifndef __ist_i3du_Camera__
#define __ist_i3du_Camera__

namespace ist {
namespace i3d {


class Viewport
{
private:
    ivec2 m_pos;
    uvec2 m_size;

public:
    Viewport() : m_pos(0,0), m_size(100,100) {}
    Viewport(const ivec2 pos, const uvec2 &size) : m_pos(pos), m_size(size) {}

    const ivec2& getPosition() const{ return m_pos; }
    const uvec2& getSize() const    { return m_size; }
    float32 getAspectRatio() const  { return (float32)m_size.x/(float32)m_size.y; }
    void setPosition(const ivec2 v) { m_pos=v; }
    void setSize(const ivec2 v)     { m_size=v; }

    bool bind() const;
};


class __declspec(align(16)) Camera
{
private:
    mat4 m_mv_matrix;
    vec4 m_position;
    vec4 m_target;
    vec4 m_up;

public:
    Camera()
    {
        m_position = vec4(0.0f, 0.0f, 100.0f, 1.0f);
        m_target = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        m_up = vec4(0.0f, 1.0f, 0.0f, 0.0f);
    }

    void setPosition(const vec4& v)     { m_position=v; }
    void setTarget(const vec4& v)       { m_target=v; }
    void setDirection(const vec4& v)    { m_up=v; }

    const vec4& getPosition() const     { return m_position; }
    const vec4& getTarget() const       { return m_target; }
    const vec4& getUp() const           { return m_up; }
    const mat4& getModelViewMatrix() const  { return m_mv_matrix; }

    void updateMatrix();
    bool bind() const;
};


class __declspec(align(16)) OrthographicCamera : public Camera
{
typedef Camera super;
private:
    mat4 m_p_matrix;
    mat4 m_mvp_matrix;
    float m_left;
    float m_right;
    float m_bottom;
    float m_top;
    float m_znear;
    float m_zfar;

public:
    OrthographicCamera()
        : m_left(0.0f)
        , m_right(100.0f)
        , m_bottom(100.0f)
        , m_top(0.0f)
        , m_znear(0.1f)
        , m_zfar(1000.0f)
    {}

    float getLeft() const   { return m_left; }
    float getRight() const  { return m_right; }
    float getBottom() const { return m_bottom; }
    float getTop() const    { return m_top; }
    float getZNear() const  { return m_znear; }
    float getZFar() const   { return m_zfar; }
    const mat4& getProjectionMatrix() const         { return m_p_matrix; }
    const mat4& getModelViewProjectionMatrix() const{ return m_mvp_matrix; }

    void setScreen(float l, float r, float b, float t)
    {
        m_left = l;
        m_right = r;
        m_bottom = b;
        m_top = t;
    }
    void setZNear(float v) { m_znear=v; }
    void setZFar(float v)  { m_zfar=v; }

    void updateMatrix();
    bool bind() const;
};


class __declspec(align(16)) PerspectiveCamera : public Camera
{
typedef Camera super;
private:
    mat4 m_p_matrix;
    mat4 m_mvp_matrix;
    float m_fovy;
    float m_aspect;
    float m_znear;
    float m_zfar;

public:
    PerspectiveCamera()
    : m_fovy(60.0f)
    , m_aspect(1.3333333f)
    , m_znear(0.1f)
    , m_zfar(1000.0f)
    {}

    float getFovy() const   { return m_fovy; }
    float getAspect() const { return m_aspect; }
    float getZNear() const  { return m_znear; }
    float getZFar() const   { return m_zfar; }
    const mat4& getProjectionMatrix() const         { return m_p_matrix; }
    const mat4& getModelViewProjectionMatrix() const{ return m_mvp_matrix; }

    void setFrustumParam(float fovy, float aspect, float znear, float zfar)
    {
        m_fovy = fovy;
        m_aspect = aspect;
        m_znear = znear;
        m_zfar = zfar;
    }
    void setFovy(float v)   { m_fovy=v; }
    void setAspect(float v) { m_aspect=v; }
    void setZNear(float v)  { m_znear=v; }
    void setZFar(float v)   { m_zfar=v; }

    void updateMatrix();
    bool bind() const;
};


} // namespace i3d
} // namespace ist
#endif //__ist_i3du_Camera__
