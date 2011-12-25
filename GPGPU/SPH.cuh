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

const int SPH_RIGID_GRID_DIV_SHIFT_X = 9; // 
const int SPH_RIGID_GRID_DIV_SHIFT_Y = 9; // 
const int SPH_RIGID_GRID_DIV_SHIFT_Z = 2; // 
const int SPH_RIGID_GRID_DIV_X = 1<<SPH_RIGID_GRID_DIV_SHIFT_X;
const int SPH_RIGID_GRID_DIV_Y = 1<<SPH_RIGID_GRID_DIV_SHIFT_Y;
const int SPH_RIGID_GRID_DIV_Z = 1<<SPH_RIGID_GRID_DIV_SHIFT_Z;
const int SPH_RIGID_GRID_DIV_3 = SPH_RIGID_GRID_DIV_X*SPH_RIGID_GRID_DIV_Y*SPH_RIGID_GRID_DIV_Z;

const int SPH_MAX_FLUID_PARTICLES = 65536;
const int SPH_MAX_RIGID_PARTICLES = 65536;
const int SPH_MAX_LIGHT_NUM = 16;
const int SPH_MAX_SPHERICAL_GRAVITY_NUM = 1;
const int SPH_THREAD_BLOCK_X = 256;


struct sphRigidParticle;

enum SPH_RIGID_SHAPE {
    SPH_RIGID_UNKNOWN,
    SPH_RIGID_SPHERE,
    SPH_RIGID_BOX,
    SPH_RIGID_BEAM,
    SPH_RIGID_END,
};

struct sphRigidClass
{
    union {
        struct {
            int shape;
            int num_particles;
            sphRigidParticle *particles;
        };
        float4 padding1;
    };
    union {
        struct {
            float sphere_radius;
        };
        struct {
            float4 box_size;
        };
        float4 padding2;
    };
};

struct sphRigidInstance
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

struct sphRigidParticle
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

struct sphBoundingBox
{
    float4 ur;
    float4 bl;
};
struct sphRigidSphere
{
    union {
        struct {
            int owner_handle;
        };
        float4 padding;
    };
    float4 pos_r;
    sphBoundingBox bb;
};

struct sphRigidBox
{
    union {
        struct {
            int owner_handle;
        };
        float4 padding;
    };
    float4 position;
    float4 planes[6];
    sphBoundingBox bb;
};


struct sphFluidParticle
{
    union {
        struct {
            int id;
            int alive;
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
            float inner_radus;
            float range_radus;
            float strength;
        };
        float4 padding;
    };
};

    
struct sphStates
{
    int fluid_num_particles;
    int rigid_num_particles;
    int fluid_alive_any;
    int rigid_alive_any;
};

struct sphDamageMessage
{
    EntityHandle to;
    float density;
};


void SPHInitialize();
void SPHFinalize();
void SPHUpdateFluid();
void SPHUpdateRigids(
    const thrust::host_vector<sphRigidInstance> &rigids,
    const thrust::host_vector<sphRigidSphere> &spheres,
    const thrust::host_vector<sphRigidBox> &boxes
    );

void SPHInitializeGLBuffers(int vbo_fluid, int vbo_rigids, int vbo_lightpos);
void SPHFinalizeGLBuffers();
void SPHSetRigidClass(sphRigidClass (&sphcc)[atomic::CB_END]);
const sphRigidClass* SPHGetRigidClass(int cid);
void SPHCopyToGL();
void SPHCopyDamageMessageToHost(sphDamageMessage *dst);
void SPHUpdateGravity(const thrust::host_vector<sphForcePointGravity> &pgravity);

sphStates& SPHGetStates();


} // extern "C"

#endif // __atomic_SPH_cuh__
