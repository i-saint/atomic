#ifndef __ist_i3dugl_Camera__
#define __ist_i3dugl_Camera__

namespace ist {
namespace i3dgl {


class istInterModule Camera
{
public:
    Camera()
    {
        m_position = vec4(0.0f, 0.0f, 100.0f, 1.0f);
        m_target = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        m_up = vec4(0.0f, 1.0f, 0.0f, 0.0f);
    }

    void setPosition(const vec4& v)     { m_position=v; }
    void setTarget(const vec4& v)       { m_target=v; }

    const vec4& getPosition() const     { return m_position; }
    const vec4& getTarget() const       { return m_target; }
    const vec4& getUp() const           { return m_up; }
    const mat4& getViewMatrix() const   { return m_v_matrix; }

    vec4 getDirection() const;
    void updateMatrix();

private:
    mat4 m_v_matrix;
    vec4 m_position;
    vec4 m_target;
    vec4 m_up;
};


class istInterModule OrthographicCamera : public Camera
{
typedef Camera super;
private:
    mat4 m_p_matrix;
    mat4 m_vp_matrix;
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
    const mat4& getProjectionMatrix() const     { return m_p_matrix; }
    const mat4& getViewProjectionMatrix() const { return m_vp_matrix; }

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
};


class istInterModule PerspectiveCamera : public Camera
{
typedef Camera super;
private:
    mat4 m_p_matrix;
    mat4 m_vp_matrix;
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
    const mat4& getProjectionMatrix() const     { return m_p_matrix; }
    const mat4& getViewProjectionMatrix() const { return m_vp_matrix; }

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
};


} // namespace i3d
} // namespace ist
#endif //__ist_i3dugl_Camera__
