#ifndef __atomic_Graphics_ParticleSet__
#define __atomic_Graphics_ParticleSet__

#include "ResourceID.h"

namespace atomic {

struct BloodstainParticle
{
    vec4 position;
    union {
        struct {
            float32 lifetime; // 1.0-0.0
        };
        float4 params;
    };
};
struct BloodstainParticle_IsDead
{
    bool operator()(const BloodstainParticle &bsp) const
    {
        return bsp.lifetime <= 0.0f;
    }
};

struct IndivisualParticle
{
    vec4 position;
    vec4 color;
    vec4 glow;
    float32 scale;
    float32 padding[3];
};
BOOST_STATIC_ASSERT(sizeof(IndivisualParticle)%16==0);

struct PSetParticle
{
    vec4 normal;
    vec3 position;
    int instanceid;
};
BOOST_STATIC_ASSERT(sizeof(PSetParticle)%16==0);

struct PSetInstance
{
    vec4 diffuse;
    vec4 glow;
    vec4 flash;
    float32 elapsed;
    float32 appear_radius;
    float32 padding[2];
};
BOOST_STATIC_ASSERT(sizeof(PSetInstance)%16==0);

struct PSetUpdateInfo
{
    mat4 transform;
    union {
        struct {
            PSET_RID psid;
            uint32 instanceid;
        };
        float4 padding;
    };
};


class ParticleSet
{
private:
    stl::vector<PSetParticle> m_particles;

public:
    void setData(const stl::vector<PSetParticle> &v) { m_particles=v; }
    uint32 getNumParticles() const { return m_particles.size(); }
    const PSetParticle* getParticleData() const { return &m_particles[0]; }
};

struct RigidInfo
{
    union {
        struct {
            float sphere_radius;
        };
        struct {
            float4 box_size;
        };
        struct {
            float4 beam_begin; // w = radius
            float4 beam_end;
        };
    };
};


} // namespace atomic
#endif // __atomic_Graphics_ParticleSet__
