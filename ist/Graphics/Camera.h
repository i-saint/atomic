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


class Camera
{
private:
    XMVECTOR m_position;
    XMVECTOR m_target;
    XMVECTOR m_direction;

protected:
    Camera()
    {
        m_position = XMVectorSet(0.0f, 0.0f, 100.0f, 1.0f);
        m_target = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
        m_direction = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    }

public:
    void setPosition(const XMVECTOR& v)  { m_position=v; }
    void setTarget(const XMVECTOR& v)    { m_target=v; }
    void setDirection(const XMVECTOR& v) { m_direction=v; }

    const XMVECTOR& getPosition() const  { return m_position; }
    const XMVECTOR& getTarget() const    { return m_target; }
    const XMVECTOR& getDirection() const { return m_direction; }

    bool bind() const;
};


class OrthographicCamera : public Camera
{
typedef Camera super;
private:
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
    float getZNear() const { return m_znear; }
    float getZFar() const  { return m_zfar; }

    void setScreen(float l, float r, float b, float t)
    {
        m_left = l;
        m_right = r;
        m_bottom = b;
        m_top = t;
    }
    void setZNear(float v) { m_znear=v; }
    void setZFar(float v)  { m_zfar=v; }

    bool bind() const;
};


class PerspectiveCamera : public Camera
{
typedef Camera super;
private:
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
    float getZNear() const { return m_znear; }
    float getZFar() const  { return m_zfar; }

    void set(float fovy, float aspect, float znear, float zfar)
    {
        m_fovy = fovy;
        m_aspect = aspect;
        m_znear = znear;
        m_zfar = zfar;
    }
    void setFovy(float v)   { m_fovy=v; }
    void setAspect(float v) { m_aspect=v; }
    void setZNear(float v) { m_znear=v; }
    void setZFar(float v)  { m_zfar=v; }

    bool bind() const;
};


} // namespace graphics
} // namespace ist
#endif //__ist_Graphic_Camera__
