#ifndef __ist_Graphic_Camera__
#define __ist_Graphic_Camera__

namespace ist {
namespace graphics {


class Viewport
{
private:
    int m_x;
    int m_y;
    size_t m_width;
    size_t m_height;

public:
    Viewport() : m_x(0), m_y(0), m_width(100), m_height(100) {}
    Viewport(int x, int y, size_t w, size_t h) : m_x(x), m_y(y), m_width(w), m_height(h) {}

    int getX() const        { return m_x; }
    int getY() const        { return m_y; }
    size_t getWidth() const { return m_width; }
    size_t getHeight() const{ return m_height; }
    float getAspectRatio() const { return (float)m_width/(float)m_height; }

    void setViewport(int x, int y, size_t w, size_t h)
    {
        m_x = x;
        m_y = y;
        m_width = w;
        m_height = h;
    }

    bool bind() const;
};


class __declspec(align(16)) Camera
{
private:
    mutable mat4 m_mv_matrix;
    vec4 m_position;
    vec4 m_target;
    vec4 m_up;
    mutable bool m_mv_modified;

protected:
    bool isMVModified() const { return m_mv_modified; }

public:
    Camera() : m_mv_modified(true)
    {
        m_position = vec4(0.0f, 0.0f, 100.0f, 1.0f);
        m_target = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        m_up = vec4(0.0f, 1.0f, 0.0f, 0.0f);
    }

    void setPosition(const vec4& v)     { m_position=v; m_mv_modified=true; }
    void setTarget(const vec4& v)       { m_target=v;   m_mv_modified=true; }
    void setDirection(const vec4& v)    { m_up=v;m_mv_modified=true; }

    const vec4& getPosition() const     { return m_position; }
    const vec4& getTarget() const       { return m_target; }
    const vec4& getUp() const    { return m_up; }
    const mat4& getModelViewMatrix() const;

    bool bind() const;
};


class __declspec(align(16)) OrthographicCamera : public Camera
{
typedef Camera super;
private:
    mutable mat4 m_p_matrix;
    mutable mat4 m_mvp_matrix;
    float m_left;
    float m_right;
    float m_bottom;
    float m_top;
    float m_znear;
    float m_zfar;
    mutable bool m_p_modified;

public:
    OrthographicCamera()
        : m_left(0.0f)
        , m_right(100.0f)
        , m_bottom(100.0f)
        , m_top(0.0f)
        , m_znear(0.1f)
        , m_zfar(1000.0f)
        , m_p_modified(true)
    {}

    float getLeft() const   { return m_left; }
    float getRight() const  { return m_right; }
    float getBottom() const { return m_bottom; }
    float getTop() const    { return m_top; }
    float getZNear() const { return m_znear; }
    float getZFar() const  { return m_zfar; }
    const mat4 getProjectionMatrix() const;
    const mat4 getModelViewProjectionMatrix() const;

    void setScreen(float l, float r, float b, float t)
    {
        m_left = l;
        m_right = r;
        m_bottom = b;
        m_top = t;
        m_p_modified = true;
    }
    void setZNear(float v) { m_znear=v; m_p_modified=true; }
    void setZFar(float v)  { m_zfar=v;  m_p_modified=true; }

    bool bind() const;
};


class __declspec(align(16)) PerspectiveCamera : public Camera
{
typedef Camera super;
private:
    mutable mat4 m_p_matrix;
    mutable mat4 m_mvp_matrix;
    float m_fovy;
    float m_aspect;
    float m_znear;
    float m_zfar;
    mutable bool m_p_modified;

public:
    PerspectiveCamera()
    : m_fovy(60.0f)
    , m_aspect(1.3333333f)
    , m_znear(0.1f)
    , m_zfar(1000.0f)
    , m_p_modified(true)
    {}

    float getFovy() const   { return m_fovy; }
    float getAspect() const { return m_aspect; }
    float getZNear() const { return m_znear; }
    float getZFar() const  { return m_zfar; }
    const mat4 getProjectionMatrix() const;
    const mat4 getModelViewProjectionMatrix() const;

    void setFrustumParam(float fovy, float aspect, float znear, float zfar)
    {
        m_fovy = fovy;
        m_aspect = aspect;
        m_znear = znear;
        m_zfar = zfar;
        m_p_modified = true;
    }
    void setFovy(float v)   { m_fovy=v;     m_p_modified=true; }
    void setAspect(float v) { m_aspect=v;   m_p_modified=true; }
    void setZNear(float v)  { m_znear=v;    m_p_modified=true; }
    void setZFar(float v)   { m_zfar=v;     m_p_modified=true; }

    bool bind() const;
};


} // namespace graphics
} // namespace ist
#endif //__ist_Graphic_Camera__
