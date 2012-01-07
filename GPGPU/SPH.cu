#define GLM_FORCE_CUDA

#include <GL/glew.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include <cutil.h>
#include <cutil_math.h>
#include <cutil_inline_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "SPH.cuh"
#include "SPH_internal.cuh"
#include "../Graphics/ResourceID.h"

__constant__ sphParams d_params;
#ifdef __atomic_enable_distance_field__
texture<float4, 3> d_distance_field;
cudaArray *d_distance_field_array;
#endif // __atomic_enable_distance_field__

#include "SPHRigid.cuh"
#include "SPHForce.cuh"
#include "SPHFluid.cuh"


FluidDataSet *h_fluid = NULL;
RigidDataSet *h_rigid = NULL;
ForceDataSet *h_forces = NULL;

sphStates h_states;
thrust::host_vector<sphFluidParticle>       h_fluid_append;
thrust::host_vector<sphFluidMessage>        h_fluid_message;

DeviceBufferObject h_fluid_gl;


struct _ClearParticles
{
    DeviceFluidDataSet dfd;
    _ClearParticles(const DeviceFluidDataSet& v) : dfd(v) {}

    __device__ void operator()(int i)
    {
        const float spacing = 0.009f;
        uint w = 128;
        dfd.particles[i].position = make_float4(
            spacing*(i%w) - (spacing*w*0.5f),
            spacing*((i/w)%w) + 0.6,
            spacing*(i/(w*w))+0.05f,
            0.0f);
        dfd.particles[i].velocity = make_float4(0.0f);
        dfd.particles[i].density = 0.0f;
        dfd.particles[i].energy = 100.0f;
        dfd.message[i].to = 0;
    }
};

void SPHInitialize(const sphParams &sph_params)
{
    h_fluid = new FluidDataSet();
    h_rigid = new RigidDataSet();
    h_forces = new ForceDataSet();

    {
#ifdef __atomic_enable_distance_field__
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        cudaExtent extent = make_cudaExtent(SPH_DISTANCE_FIELD_DIV_X, SPH_DISTANCE_FIELD_DIV_Y, SPH_DISTANCE_FIELD_DIV_Z);

        d_distance_field.addressMode[0] = cudaAddressModeClamp;
        d_distance_field.addressMode[1] = cudaAddressModeClamp;
        d_distance_field.addressMode[2] = cudaAddressModeClamp;
        d_distance_field.filterMode     = cudaFilterModeLinear;
        d_distance_field.normalized     = true;

        CUDA_SAFE_CALL( cudaMalloc3DArray(&d_distance_field_array, &desc, extent) );
        CUDA_SAFE_CALL( cudaBindTextureToArray(d_distance_field, d_distance_field_array, desc) );
#endif __atomic_enable_distance_field__
    }
    {
        CUDA_SAFE_CALL( cudaMemcpyToSymbol("d_params", &sph_params, sizeof(sph_params)) );

        sphGridParam grid_params;
        const float grid_len = 5.12f;
        grid_params.grid_dim = make_float4(grid_len, grid_len, sph_params.smooth_len*SPH_FLUID_GRID_DIV_Z, 0.0f);
        grid_params.grid_dim_rcp = make_float4(1.0f) / (grid_params.grid_dim / make_float4(SPH_FLUID_GRID_DIV_X, SPH_FLUID_GRID_DIV_Y, SPH_FLUID_GRID_DIV_Z, 1.0));
        grid_params.grid_pos = make_float4(-grid_len/2.0f, -grid_len/2.0f, 0.0f, 0.0f);
        h_fluid->params[0] = grid_params;

        h_states.fluid_num_particles = 30000;
        h_states.fluid_alive_any = 0;
        h_fluid->states[0] = h_states;
    }

    h_fluid->resizeParticles(h_states.fluid_num_particles);
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(h_states.fluid_num_particles), _ClearParticles(h_fluid->getDeviceData()) );
}

void SPHFinalize()
{
#ifdef __atomic_enable_distance_field__
        CUDA_SAFE_CALL( cudaUnbindTexture(d_distance_field) );
        CUDA_SAFE_CALL( cudaFreeArray(d_distance_field_array) );
#endif // __atomic_enable_distance_field__
    delete h_forces;h_forces=NULL;
    delete h_rigid; h_rigid=NULL;
    delete h_fluid; h_fluid=NULL;
}


void SPHInitializeGLBuffers(int vbo_fluid)
{
    h_fluid_gl.registerBuffer(vbo_fluid, cudaGraphicsMapFlagsWriteDiscard);
}

void SPHFinalizeGLBuffers()
{
    h_fluid_gl.unregisterBuffer();
}


#ifdef __atomic_enable_distance_field__
void SPHUpdateDistanceField(const float4 *distances, const EntityHandle *entities)
{
    cudaExtent extent = make_cudaExtent(SPH_DISTANCE_FIELD_DIV_X, SPH_DISTANCE_FIELD_DIV_Y, SPH_DISTANCE_FIELD_DIV_Z);
    cudaMemcpy3DParms params = {0};
    params.srcPtr   = make_cudaPitchedPtr((void*)distances, extent.width*sizeof(float4), extent.width, extent.height);
    params.dstArray = d_distance_field_array;
    params.extent   = extent;
    params.kind     = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&params) );

    thrust::copy(entities, entities+SPH_DISTANCE_FIELD_DIV_3, h_fluid->df_entities.begin());
}
#endif // __atomic_enable_distance_field__

void SPHUpdateRigids(
    const thrust::host_vector<sphRigidPlane> &planes,
    const thrust::host_vector<sphRigidSphere> &spheres,
    const thrust::host_vector<sphRigidBox> &boxes
    )
{
    h_rigid->planes = planes;
    h_rigid->spheres = spheres;
    h_rigid->boxes = boxes;
}


void SPHUpdateForce(const thrust::host_vector<sphForcePointGravity> &pgravity)
{
    h_forces->point_gravity = pgravity;
}

void SPHAddFluid(const thrust::host_vector<sphFluidParticle> &particles)
{
    if(particles.empty()) { return; }

    uint current = h_fluid->particles.size();
    h_fluid->resizeParticles(current + particles.size());
    thrust::copy(particles.begin(), particles.end(), h_fluid->particles.begin()+current);
    h_states.fluid_num_particles = h_fluid->particles.size();
    h_fluid->states[0] = h_states;
}

void SPHUpdateFluid()
{
    DeviceFluidDataSet dfd = h_fluid->getDeviceData();
    DeviceRigidDataSet drd = h_rigid->getDeviceData();
    DeviceForceDataSet dgd = h_forces->getDeviceData();
    int num_particles = (int)h_fluid->particles.size();

    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidUpdateHash(dfd) );
    thrust::sort_by_key(h_fluid->hashes.begin(), h_fluid->hashes.end(), h_fluid->particles.begin());
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(SPH_FLUID_GRID_DIV_3), _FluidClearGrid(dfd));
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidUpdateGrid(dfd));

    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidComputeDensity(dfd));
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidComputeForce(dfd));
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidIntegrate(dfd, drd, dgd));
    h_fluid_message = h_fluid->message;

    sphStates stat = h_fluid->states[0];
    h_states.fluid_num_particles = stat.fluid_alive_any==0 ? 0 : stat.fluid_num_particles;
    h_states.fluid_alive_any = 0;
    h_fluid->states[0] = h_states;
    h_fluid->resizeParticles(stat.fluid_num_particles);
}


void SPHCopyToGL()
{
    sphFluidParticle *gl_fluid = (sphFluidParticle*)h_fluid_gl.mapBuffer();
    thrust::copy(h_fluid->particles.begin(), h_fluid->particles.end(), thrust::device_ptr<sphFluidParticle>(gl_fluid));
    h_fluid_gl.unmapBuffer();
}


const sphStates& SPHGetStates()
{
    return h_states;
}

const thrust::host_vector<sphFluidMessage>& SPHGetFluidMessage()
{
    return h_fluid_message;
}
