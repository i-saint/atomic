#ifndef __atomic_Graphics_Light__
#define __atomic_Graphics_Light__
namespace atomic {

const int ATOMIC_MAX_DIRECTIONAL_LIGHTS = 16;
const int ATOMIC_MAX_POINT_LIGHTS = 256;

struct DirectionalLight
{
    vec4 direction;
    vec4 diffuse_color;
    vec4 ambient_color;
};
BOOST_STATIC_ASSERT(sizeof(DirectionalLight)==sizeof(vec4)*3);

struct PointLight
{
    vec4 position;
    vec4 color;
    vec4 direction;
    float min_range;
    float max_range;
    float padding[2];
};
BOOST_STATIC_ASSERT(sizeof(PointLight)==sizeof(vec4)*4);

} // namespace atomic
#endif // __atomic_Graphics_Light__
