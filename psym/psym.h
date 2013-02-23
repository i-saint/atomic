#ifndef SPH_types_h
#define SPH_types_h

#include <vector>
#include "ist/Base.h"
#include "psymConst.h"
#include "psymCore_ispc.h"
#include "psymSoA.h"
#include "../DynamicObjLoader/DynamicObjLoader.h"

#define psymGlobalNamespace(...) } __VA_ARGS__ namespace psym {

namespace psym {

#define set_xyz(v, _x, _y, _z)  v.x =_x; v.y =_y; v.z =_z;
#define set_nxyz(v, _x, _y, _z) v.nx=_x; v.ny=_y; v.nz=_z;
#define set_vxyz(v, _x, _y, _z) v.vx=_x; v.vy=_y; v.vz=_z;

typedef short           int16;
typedef unsigned short  uint16;
typedef int             int32;
typedef unsigned int    uint32;
typedef float           float32;

using ispc::Particle_SOA8;
using ispc::GridData;

using ispc::RigidSphere;
using ispc::RigidPlane;
using ispc::RigidBox;

using ispc::PointForce;
using ispc::DirectionalForce;
using ispc::BoxForce;


struct istAlign(16) Particle
{
    simdvec4 position;
    simdvec4 velocity;
    struct {
        float32 energy;
        float32 density;
        uint32 hash;
        uint32 hit_to;
    };
};
psymGlobalNamespace(
    istSerializeRaw(psym::Particle);
    )



class istAlign(16) World
{
public:
    World();
    void update(float32 dt);

    void clearRigidsAndForces();

    void addRigid(const RigidSphere &v);
    void addRigid(const RigidPlane &v);
    void addRigid(const RigidBox &v);
    void addForce(const PointForce &v);
    void addForce(const DirectionalForce &v);
    void addForce(const BoxForce &v);
    void addParticles(const Particle *p, size_t num_particles);

    const Particle* getParticles() const;
    size_t getNumParticles() const;

public:
    Particle particles[PSYM_MAX_PARTICLE_NUM]; // need serialize

    Particle_SOA8 particles_soa[PSYM_MAX_PARTICLE_NUM];
    GridData cell[PSYM_GRID_DIV][PSYM_GRID_DIV];

    size_t num_active_particles; // need serialize

    ist::raw_vector<RigidSphere>   collision_spheres;
    ist::raw_vector<RigidPlane>    collision_planes;
    ist::raw_vector<RigidBox>      collision_boxes;

    ist::raw_vector<PointForce>       force_point;
    ist::raw_vector<DirectionalForce> force_directional;
    ist::raw_vector<BoxForce>         force_box;


    istSerializeBlock({
        ar & num_active_particles;
        for(size_t i=0; i<num_active_particles; ++i) {
            ar & particles[i];
        }
    })
};

} // namespace psym
#endif // SPH_types_h
