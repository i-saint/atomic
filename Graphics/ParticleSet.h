#ifndef atm_Graphics_ParticleSet_h
#define atm_Graphics_ParticleSet_h

#include "ResourceID.h"

namespace atm {

struct istAlign(16) BloodstainParticle
{
    vec4 position;
    union {
        struct {
            float32 lifetime; // 1.0-0.0
        };
        float32 params[4];
    };
};
atmSerializeRaw(BloodstainParticle);

struct istAlign(16) SingleParticle
{
    vec4 position;
    vec4 color;
    vec4 glow;
    float32 scale;
    float32 padding[3];
};
istStaticAssert(sizeof(SingleParticle)%16==0);

struct istAlign(16) PSetParticle
{
    vec4 normal;
    vec3 position;
    int instanceid;
};
istStaticAssert(sizeof(PSetParticle)%16==0);

struct istAlign(16) PSetInstance
{
    vec4 diffuse;
    vec4 glow;
    vec4 flash;
    float32 elapsed;
    float32 appear_radius;
    float32 scale;
    float32 padding[1];
    mat4 transform;
    mat4 rotate;

    PSetInstance() : elapsed(0.0f), appear_radius(100.0f), scale(1.0f)
    {}
};
istStaticAssert(sizeof(PSetInstance)%16==0);

struct istAlign(16) PSetUpdateInfo
{
    union {
        struct {
            PSET_RID psid;
            uint32 instanceid;
            uint32 num;
        };
        float padding[4];
    };
};


class istAlign(16) ParticleSet
{
private:
    AABB m_aabb;
    ist::vector<PSetParticle> m_particles;

public:
    void setData(const ist::vector<PSetParticle> &v)
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

struct PSetDrawData
{
    ist::vector<PSetUpdateInfo> update_info;
    ist::vector<PSetInstance>   instance_data;
    ist::vector<PSetParticle>   particle_data;
    VBO_RID     vbo;
    SH_RID      shader;
    TEX2D_RID   params;

    PSetDrawData() : vbo(), shader(), params() {}
    void clear()
    {
        update_info.clear();
        instance_data.clear();
        particle_data.clear();
    }
};


struct istAlign(16) RigidInfo
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

struct istAlign(16) ModelInfo
{
    I3D_TOPOLOGY topology;
    VA_RID vertices;
    IBO_RID indices;
    uint32 num_indices;

    ModelInfo(I3D_TOPOLOGY t=I3D_POINTS, VA_RID v=VA_NULL, IBO_RID i=IBO_NULL, uint32 n=0)
        : topology(t), vertices(v), indices(i), num_indices(n)
    {}
};

} // namespace atm

#endif // atm_Graphics_ParticleSet_h
