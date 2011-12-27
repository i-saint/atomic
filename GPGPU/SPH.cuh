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
typedef uint sphHash;
typedef uint2 sphGridData;
typedef int sphDeadFlag;
typedef glm::vec4 vec4;
typedef glm::mat4 mat4;


const int SPH_FLUID_GRID_DIV_SHIFT_X = 9; // 
const int SPH_FLUID_GRID_DIV_SHIFT_Y = 9; // 
const int SPH_FLUID_GRID_DIV_SHIFT_Z = 2; // 
const int SPH_FLUID_GRID_DIV_X = 1<<SPH_FLUID_GRID_DIV_SHIFT_X;
const int SPH_FLUID_GRID_DIV_Y = 1<<SPH_FLUID_GRID_DIV_SHIFT_Y;
const int SPH_FLUID_GRID_DIV_Z = 1<<SPH_FLUID_GRID_DIV_SHIFT_Z;
const int SPH_FLUID_GRID_DIV_3 = SPH_FLUID_GRID_DIV_X*SPH_FLUID_GRID_DIV_Y*SPH_FLUID_GRID_DIV_Z;

const int SPH_MAX_FLUID_PARTICLES = 65536 * 4;
const int SPH_MAX_RIGID_PARTICLES = 65536 * 4;
const int SPH_MAX_LIGHT_NUM = 16;
const int SPH_MAX_POINT_GRAVITY_NUM = 1;
const int SPH_THREAD_BLOCK_X = 256;


struct sphRigidParticle;

enum SPH_RIGID_SHAPE {
    SPH_RIGID_UNKNOWN,
    SPH_RIGID_SPHERE,
    SPH_RIGID_BOX,
    SPH_RIGID_BEAM,
    SPH_RIGID_END,
};

struct sphBoundingBox
{
    float4 ur;
    float4 bl;
};

struct sphRigidCollision
{
    union {
        struct {
            int shape; // SPH_RIGID_SHAPE
            int owner_handle;
        };
        float4 padding;
    };
    sphBoundingBox bb;
};

struct sphRigidSphere : public sphRigidCollision
{
    float4 pos_r;
};

struct sphRigidBox : public sphRigidCollision
{
    float4 position;
    float4 planes[6];
};


struct sphFluidParticle
{
    union {
        struct {
            float energy;
            float density;
            EntityHandle owner;
        };
        float4 padding;
    };
    float4 position;
    float4 velocity;
};

struct sphFluidForce
{
    float4 acceleration;
    union {
        struct {
            float density;
        };
        float4 padding;
    };
};


struct sphForcePointGravity
{
    float4 position;
    union {
        struct {
            float radius;
            float strength;
        };
        float4 padding;
    };
};

    
struct sphStates
{
    int fluid_num_particles;
    int fluid_alive_any;
};

struct sphFluidMessage
{
    union {
        float4 velocity;
        struct {
            float3 velocity3;
            EntityHandle to;
        };
    };
};


void SPHInitialize();
void SPHFinalize();
void SPHUpdateRigids(
    const thrust::host_vector<sphRigidSphere> &spheres,
    const thrust::host_vector<sphRigidBox> &boxes
    );
void SPHUpdateForce(
    const thrust::host_vector<sphForcePointGravity> &pgravity
    );
void SPHAddFluid(const thrust::host_vector<sphFluidParticle> &particles);
void SPHUpdateFluid();

void SPHInitializeGLBuffers(int vbo_fluid);
void SPHFinalizeGLBuffers();
void SPHCopyToGL();

const sphStates& SPHGetStates();
const thrust::host_vector<sphFluidMessage>& SPHGetFluidMessage();


} // extern "C"

#endif // __atomic_SPH_cuh__
