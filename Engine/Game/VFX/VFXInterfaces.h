#ifndef atm_Engine_Game_VFX_VFXInterfaces_h
#define atm_Engine_Game_VFX_VFXInterfaces_h
namespace atm {

struct VFXScintillaSpawnData
{
    vec4 color;
    vec4 glow;
    vec3 position;
    vec3 velosity;
    float32 scale;
    float32 lifetime;
    float32 scatter_radius;
    float32 diffuse_strength;
    uint32 num_particles;

    VFXScintillaSpawnData()
        : scale(1.00f), lifetime(0.0f), scatter_radius(0.02f), diffuse_strength(0.01f), num_particles(0)
    {}
};
void VFXScintillaSpawn(const VFXScintillaSpawnData &v);


struct VFXLightSpawnData
{
    vec4    color;
    vec3    position;
    float32 radius;
    vec4    color_attenuation;
    float32 radius_attenuation;

    VFXLightSpawnData()
        : radius(0.5f), radius_attenuation(0.01f)
    {}
};
void VFXLightSpawn(const VFXLightSpawnData &v);


struct VFXShockwaveSpawnData
{
    vec3 position;
    float32 radius;
    float32 speed;
    float32 lifetime;

    VFXShockwaveSpawnData()
        : radius(0.1f), speed(0.1f), lifetime(20.0f)
    {}
};
void VFXShockwaveSpawn(const VFXShockwaveSpawnData &v);


struct VFXFeedbackBlurSpawnData
{
    vec3 position;
    float32 radius;
    float32 speed;
    float32 lifetime;

    VFXFeedbackBlurSpawnData()
        : radius(0.1f), speed(0.1f), lifetime(20.0f)
    {}
};
void VFXFeedbackBlurSpawn(const VFXFeedbackBlurSpawnData &v);

class IVFXComponent;
class VFXScintilla;
class VFXLight;
class VFXShockwave;
class VFXFeedbackBlur;
IVFXComponent* VFXScintillaCreate();
IVFXComponent* VFXLightCreate();
IVFXComponent* VFXShockwaveCreate();
IVFXComponent* VFXFeedbackBlurCreate();

} // namespace atm
#endif // atm_Engine_Game_VFX_VFXInterfaces_h
