#ifndef atomic_Graphics_ParticleSet_h
#define atomic_Graphics_ParticleSet_h

#include "ResourceID.h"

namespace atomic {

struct BloodstainParticle
{
    vec4 position;
    union {
        struct {
            float32 lifetime; // 1.0-0.0
        };
        float params[4];
    };
};
atomicInterruptNamespace(
    istSerializeRaw(atomic::BloodstainParticle);
)

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
    mat4 translate;
};
BOOST_STATIC_ASSERT(sizeof(PSetInstance)%16==0);

struct PSetUpdateInfo
{
    union {
        struct {
            PSET_RID psid;
            uint32 instanceid;
        };
        float padding[4];
    };
};


class ParticleSet
{
private:
    stl::vector<PSetParticle> m_particles;
    AABB m_aabb;

public:
    void setData(const stl::vector<PSetParticle> &v)
    {
        m_particles = v;
        simdvec4 bb_min = simdvec4( 9999.0f,  9999.0f,  9999.0f, 1.0f);
        simdvec4 bb_max = simdvec4(-9999.0f, -9999.0f, -9999.0f, 1.0f);
        for(size_t i=0; i<v.size(); ++i) {
            simdvec4 pos = simdvec4(v[i].position, 1.0f);
            bb_min = glm::min(bb_min, pos);
            bb_max = glm::max(bb_max, pos);
        }
        m_aabb = AABB(bb_min.Data, bb_max.Data);
    }
    uint32 getNumParticles() const { return m_particles.size(); }
    const PSetParticle* getParticleData() const { return &m_particles[0]; }
    const AABB& getAABB() const { return m_aabb; }
};

struct RigidInfo
{
    union {
        struct {
            float sphere_radius;
        };
        struct {
            float box_size[4];
        };
        struct {
            float beam_begin[4]; // w = radius
            float beam_end[4];
        };
    };
};

} // namespace atomic

#endif // atomic_Graphics_ParticleSet_h
