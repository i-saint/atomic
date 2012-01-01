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

__constant__ sphParam d_params;

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

void SPHInitialize()
{
    h_fluid = new FluidDataSet();
    h_rigid = new RigidDataSet();
    h_forces = new ForceDataSet();

    {
        sphParam sph_params;
        sph_params.smooth_len           = 0.02f;
        sph_params.pressure_stiffness   = 200.0f;
        sph_params.rest_density         = 1000.0f;
        sph_params.particle_mass        = 0.001f;
        sph_params.viscosity            = 0.1f;
        sph_params.density_coef         = sph_params.particle_mass * 315.0f / (64.0f * CUDART_PI_F * pow(sph_params.smooth_len, 9));
        sph_params.grad_pressure_coef   = sph_params.particle_mass * -45.0f / (CUDART_PI_F * pow(sph_params.smooth_len, 6));
        sph_params.lap_viscosity_coef   = sph_params.particle_mass * sph_params.viscosity * 45.0f / (CUDART_PI_F * pow(sph_params.smooth_len, 6));
        sph_params.wall_stiffness       = 3000.0f;
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
