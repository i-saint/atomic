#ifndef _SPH_types_h_
#define _SPH_types_h_

#include "SPH_const.h"
#include "SPH_core_ispc.h"
#include "SoA.h"
typedef int int32;
typedef unsigned int uint32;
typedef float float32;

using ist::simdvec4;
using ist::simdvec8;
using ist::soavec24;
using ist::soavec34;
using ist::soavec44;

typedef ispc::Particle_SOA8 sphParticleSOA8;
typedef ispc::GridData sphGridData;


__declspec(align(16)) 
struct sphParticle
{
    simdvec4 position;
    simdvec4 velocity;
    union {
        struct {
            float32 density;
            uint32 hash;
            uint32 dummy[2];
        } params;
        simdvec4 paramsv;
    };
};

__declspec(align(16)) 
struct sphGrid
{
    sphParticle particles[SPH_MAX_PARTICLE_NUM];
    sphParticleSOA8 particles_soa[SPH_MAX_PARTICLE_NUM];
    sphGridData cell[SPH_GRID_DIV][SPH_GRID_DIV];

    sphGrid();
    void update();
};



#endif // _SPH_types_h_
