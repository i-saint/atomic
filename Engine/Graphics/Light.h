#ifndef atm_Engine_Graphics_Light_h
#define atm_Engine_Graphics_Light_h
namespace atm {

const int ATOMIC_MAX_DIRECTIONAL_LIGHTS = 16;
const int ATOMIC_MAX_POINT_LIGHTS = 1024;

class DirectionalLight
{
private:
    vec4 m_direction;
    vec4 m_diffuse;
    vec4 m_ambient;

public:
    void setDirection(const vec3 &v){ m_direction=vec4(v,0.0f); }
    void setDiffuse(const vec4 &v)  { m_diffuse=v; }
    void setAmbient(const vec4 &v)  { m_ambient=v; }
    const vec3  getDirection() const{ return vec3(m_direction); }
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
        float padding[4];
    };

public:
    void setPosition(const vec3 &v) { m_position=vec4(v,1.0f); }
    void setColor(const vec4 &v)    { m_color=v; }
    void setRadius(float32 v)       { m_radius=v; m_rcp_radius=1.0f/v; }

    const vec3 getPosition() const  { return vec3(m_position); }
    const vec4& getColor() const    { return m_color; }
    float32 getRadius() const       { return m_radius; }
};
BOOST_STATIC_ASSERT(sizeof(PointLight)==sizeof(vec4)*3);

} // namespace atm
#endif // atm_Engine_Graphics_Light_h
