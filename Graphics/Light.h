#ifndef __atomic_Graphics_Light__
#define __atomic_Graphics_Light__
namespace atomic {

const int ATOMIC_MAX_DIRECTIONAL_LIGHTS = 16;
const int ATOMIC_MAX_POINT_LIGHTS = 256;

class DirectionalLight
{
private:
    vec4 m_direction;
    vec4 m_diffuse;
    vec4 m_ambient;

public:
    void setDirection(const vec4 &v){ m_direction=v; }
    void setDiffuse(const vec4 &v)  { m_diffuse=v; }
    void setAmbient(const vec4 &v)  { m_ambient=v; }
    const vec4& getDirection() const{ return m_direction; }
    const vec4& getDiffuse() const  { return m_diffuse; }
    const vec4& getAmbient() const  { return m_ambient; }
};
BOOST_STATIC_ASSERT(sizeof(DirectionalLight)==sizeof(vec4)*3);

class PointLight
{
private:
    vec4 m_position;
    vec4 m_color;
    union {
        struct {
            float m_radius;
            float m_rcp_radius;
        };
        float4 padding;
    };

public:
    void setPosition(const vec4 &v) { m_position=v; }
    void setColor(const vec4 &v)    { m_color=v; }
    void setRadius(float32 v)       { m_radius=v; m_rcp_radius=1.0f/v; }

    const vec4& getPosition() const { return m_position; }
    const vec4& getColor() const    { return m_color; }
    float32 getRadius() const       { return m_radius; }
};
BOOST_STATIC_ASSERT(sizeof(PointLight)==sizeof(vec4)*3);

} // namespace atomic
#endif // __atomic_Graphics_Light__
