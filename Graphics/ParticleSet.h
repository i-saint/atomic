#ifndef __atomic_Graphics_ParticleSet__
#define __atomic_Graphics_ParticleSet__

#include "ResourceID.h"

namespace atomic {

struct PSetParticle
{
    vec4 position;
    vec4 normal;
    vec4 diffuse;
    vec4 glow;
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

struct PSetInstance
{
    mat4 transform;
    vec4 diffuse;
    vec4 glow;
    union {
        struct {
            PSET_RID psid;
        };
        float4 padding;
    };
};


} // namespace atomic
#endif // __atomic_Graphics_ParticleSet__
