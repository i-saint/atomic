#ifndef __atomic_SPH_cuh__
#define __atomic_SPH_cuh__

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../Graphics/ResourceID.h"

extern "C" {

typedef unsigned int uint;
typedef uint EntityHandle;
typedef uint SPHHash;
typedef uint2 SPHGridData;
typedef glm::vec4 vec4;
typedef glm::mat4 mat4;


const int SPH_GRID_DIV_SHIFT_X = 9; // 
const int SPH_GRID_DIV_SHIFT_Y = 9; // 
const int SPH_GRID_DIV_SHIFT_Z = 2; // 
const int SPH_GRID_DIV_X = 1<<SPH_GRID_DIV_SHIFT_X;
const int SPH_GRID_DIV_Y = 1<<SPH_GRID_DIV_SHIFT_Y;
const int SPH_GRID_DIV_Z = 1<<SPH_GRID_DIV_SHIFT_Z;
const int SPH_GRID_DIV_3 = SPH_GRID_DIV_X*SPH_GRID_DIV_Y*SPH_GRID_DIV_Z;
//const int SPH_MAX_FLUID_PARTICLES = 65536*2;
const int SPH_MAX_FLUID_PARTICLES = 65536;
const int SPH_MAX_RIGID_PARTICLES = 65536;
const int SPH_MAX_LIGHT_NUM = 16;
const int SPH_MAX_SPHERICAL_GRAVITY_NUM = 1;
const int SPH_THREAD_BLOCK_X = 256;


struct SPHFluidParticle
{
    union {
        struct {
            int id;
            int lifetime;
            float density;
            EntityHandle owner;
        };
        float4 padding;
    };
    float4 position;
    float4 velocity;
};

struct SPHFluidParticleForce
{
    float4 acceleration;
    union {
        struct {
            float density;
        };
        float4 padding;
    };
};

struct SPHRigidParticle
{
    union {
        struct {
            int owner_handle;
        };
        float4 padding;
    };
    float4 position;
    float4 normal;
};

struct SPHSphericalGravity
{
    float4 position;
    union {
        struct {
            int is_active;
            float inner_radus;
            float range_radus;
            float strength;
        };
        float4 padding;
    };
};


struct SPHRigidClass
{
    union {
        struct {
            int num_particles;
            SPHRigidParticle *particles;
        };
        float4 padding;
    };
};

struct SPHRigidInstance
{
    union {
        struct {
            EntityHandle handle;
            int classid;
        };
        float4 padding;
    };
    glm::mat4 transform;
};
    
struct SPHGPUStates
{
    int num_particles;
};

struct SPHDamageMessage
{
    EntityHandle to;
    float damage;
};


void SPHInitialize();
void SPHFinalize();
void SPHUpdateFluid();
void SPHUpdateRigids(const thrust::host_vector<SPHRigidInstance> &rigids);

void SPHInitializeGLBuffers(int vbo_fluid, int vbo_rigids, int vbo_lightpos);
void SPHFinalizeGLBuffers();
void SPHCopyRigidClassInfo(SPHRigidClass (&sphcc)[atomic::CB_END]);
void SPHCopyToGL();
void SPHCopyDamageMessageToHost(SPHDamageMessage *dst);

void SPHUpdateGravity(SPHSphericalGravity (&sgravity)[ SPH_MAX_SPHERICAL_GRAVITY_NUM ]);

void SPHSpawnFluidParticles(const thrust::host_vector<SPHRigidInstance> &rigids);


} // extern "C"

#endif // __atomic_SPH_cuh__
