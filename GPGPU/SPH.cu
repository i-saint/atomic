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



__global__ void GClearParticles(DeviceFluidDataSet ps)
{
    const float spacing = 0.009f;
    int i = GetThreadId();
    ps.particles[i].id = i;
    ps.particles[i].alive = 0xffffffff;
    uint w = 128;
    ps.particles[i].position = make_float4(
        spacing*(i%w) - (spacing*w*0.5f),
        spacing*((i/w)%w) + 0.6,
        spacing*(i/(w*w))+0.05f,
        0.0f);
    ps.particles[i].velocity = make_float4(0.0f);

    ps.forces[i].density = 0.0f;
    ps.forces[i].acceleration = make_float4(0.0f);
}

void SPHInitialize()
{
    h_fluid = new FluidDataSet();
    h_rigid = new RigidDataSet();
    h_forces = new ForceDataSet();

    {
        sphParam sph_params;
        sph_params.smooth_len          = 0.02f;
        sph_params.pressure_stiffness  = 200.0f;
        sph_params.rest_density        = 1000.0f;
        sph_params.particle_mass       = 0.001f;
        sph_params.viscosity           = 0.1f;
        sph_params.density_coef        = sph_params.particle_mass * 315.0f / (64.0f * CUDART_PI_F * pow(sph_params.smooth_len, 9));
        sph_params.grad_pressure_coef  = sph_params.particle_mass * -45.0f / (CUDART_PI_F * pow(sph_params.smooth_len, 6));
        sph_params.lap_viscosity_coef  = sph_params.particle_mass * sph_params.viscosity * 45.0f / (CUDART_PI_F * pow(sph_params.smooth_len, 6));
        sph_params.wall_stiffness      = 3000.0f;
        CUDA_SAFE_CALL( cudaMemcpyToSymbol("d_params", &sph_params, sizeof(sph_params)) );

        sphGridParam grid_params;
        const float grid_len = 5.12f;
        grid_params.grid_dim = make_float4(grid_len, grid_len, sph_params.smooth_len*SPH_FLUID_GRID_DIV_Z, 0.0f);
        grid_params.grid_dim_rcp = make_float4(1.0f) / (grid_params.grid_dim / make_float4(SPH_FLUID_GRID_DIV_X, SPH_FLUID_GRID_DIV_Y, SPH_FLUID_GRID_DIV_Z, 1.0));
        grid_params.grid_pos = make_float4(-grid_len/2.0f, -grid_len/2.0f, 0.0f, 0.0f);
        h_fluid->params[0] = grid_params;

        sphStates stat;
        stat.num_fluid_particles = SPH_MAX_FLUID_PARTICLES;
        stat.num_rigid_particles = 0;
        h_fluid->states[0] = stat;
    }
    {
        sphSphericalGravity h_sg;
        h_sg.position = make_float4(0.0f);
        h_sg.is_active = 1;
        h_sg.inner_radus = 0.5f;
        h_sg.range_radus = 5.12f;
        h_sg.strength = 0.5f;
        h_forces->sgravities[0] = h_sg;
    }

    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_FLUID_PARTICLES / SPH_THREAD_BLOCK_X );
    GClearParticles<<<dimGrid, dimBlock>>>(h_fluid->getDeviceData());
}

void SPHFinalize()
{
    delete h_forces;h_forces=NULL;
    delete h_rigid; h_rigid=NULL;
    delete h_fluid; h_fluid=NULL;
}




struct _FluidIntegrate
{
    DeviceFluidDataSet dfd;
    DeviceForceDataSet dgd;
    _FluidIntegrate(const DeviceFluidDataSet &v, const DeviceForceDataSet &g) : dfd(v), dgd(g) {}

    __device__ void operator()(int i)
    {
        const uint P_ID = i;

        float4 position = dfd.particles[P_ID].position;
        float4 velocity = dfd.particles[P_ID].velocity;
        float4 acceleration = dfd.forces[P_ID].acceleration;
        dfd.hashes[P_ID] = 0;

        //const float3 planes[4] = {
        //    make_float3( 1.0f, 0.0f, 0),
        //    make_float3( 0.0f, 1.0f, 0),
        //    make_float3(-1.0f, 0.0f, 2.56f),
        //    make_float3( 0.0f,-1.0f, 2.56f),
        //};
        //// Apply the forces from the map walls
        //for(uint i = 0 ; i < 4 ; i++)
        //{
        //    float dist = dot(make_float3(position.x, position.y, 1.0f), planes[i]);
        //    acceleration += min(dist, 0.0f) * -d_param.wall_stiffness * make_float4(planes[i].x, planes[i].y, 0.0f, 0.0f);
        //}
        //float4 gravity = make_float4(0.0f, -0.5f, 0.0f, 0.0f);

        acceleration += min(position.z, 0.0f) * -d_params.wall_stiffness * make_float4(0.0f, 0.0f, 0.5f, 0.0f);
        acceleration += make_float4(0.0f, 0.0f, -5.0f, 0.0f);


        // Apply gravity
        for(int i=0; i<SPH_MAX_SPHERICAL_GRAVITY_NUM; ++i) {
            if(!dgd.sgravity[i].is_active) { continue; }

            const float4 center = dgd.sgravity[i].position;
            const float gravity_strength = dgd.sgravity[i].strength;
            const float inner_radius = dgd.sgravity[i].inner_radus;
            const float outer_radius = dgd.sgravity[i].range_radus;

            float4 diff = center-position;
            diff.w = 0.0f;
            float distance = length(diff);
            float4 dir = diff/distance;
            float4 gravity = dir * gravity_strength;

            acceleration += min(distance-inner_radius, 0.0f) * d_params.wall_stiffness * dir;
            acceleration += min(outer_radius-distance, 0.0f) * -d_params.wall_stiffness * dir;
            acceleration += gravity;

            //// kill
            //if(distance-inner_radius < 0.0f) { dfd.hashes[P_ID] = 1; }
        }

        //const float timestep = 1.0f/60.f;
        const float timestep = 0.01f;

        // Integrate
        velocity += timestep * acceleration;
        velocity *= make_float4(0.999);
        if(dot(velocity, velocity) > 1.0f) { velocity *= make_float4(0.98); }
        //velocity.z *= 0.0f;
        position += timestep * velocity;
        //position.z *= 0.0f;

        // Update
        dfd.particles[P_ID].density = dfd.forces[P_ID].density;
        dfd.particles[P_ID].position = position;
        dfd.particles[P_ID].velocity = velocity;
    }
};

void SPHUpdateFluid()
{
    DeviceFluidDataSet dfd = h_fluid->getDeviceData();
    DeviceForceDataSet dgd = h_forces->getDeviceData();
    int num_particles = (int)h_fluid->particles.size();

    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidUpdateHash(dfd) );
    thrust::sort_by_key(h_fluid->hashes.begin(), h_fluid->hashes.end(), h_fluid->particles.begin());
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(SPH_FLUID_GRID_DIV_3), _FluidGridClear(dfd));
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidGridUpdate(dfd));

    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidComputeDensity(dfd));
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidComputeForce(dfd));
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidIntegrate(dfd, dgd));
    thrust::sort_by_key(h_fluid->hashes.begin(), h_fluid->hashes.end(), h_fluid->particles.begin());
    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_particles), _FluidCountAlives(dfd));

    const sphStates &stat = h_fluid->states[0];
    h_states.num_fluid_particles = stat.num_fluid_particles;
    if(h_states.num_fluid_particles==1) {
        h_states.num_fluid_particles = h_fluid->hashes[0]==1 ? 1 : 0;
        h_fluid->states[0] = h_states;
    }
    h_fluid->resizeParticles(stat.num_fluid_particles);
}




DeviceBufferObject h_fluid_gl;
DeviceBufferObject h_rigids_gl;
DeviceBufferObject h_light_gl;
thrust::host_vector<sphRigidClass>          h_rigid_class;
thrust::host_vector<sphRigidUpdateInfo>     h_rigid_ui;


void SPHInitializeGLBuffers(int vbo_fluid, int vbo_rigids, int vbo_lightpos)
{
    h_fluid_gl.registerBuffer(vbo_fluid, cudaGraphicsMapFlagsWriteDiscard);
    h_rigids_gl.registerBuffer(vbo_rigids, cudaGraphicsMapFlagsWriteDiscard);
    h_light_gl.registerBuffer(vbo_lightpos, cudaGraphicsMapFlagsWriteDiscard);
}

void SPHFinalizeGLBuffers()
{
    h_light_gl.unregisterBuffer();
    h_rigids_gl.unregisterBuffer();
    h_fluid_gl.unregisterBuffer();
}

void SPHCopyRigidClassInfo(sphRigidClass (&sphcc)[atomic::CB_END])
{
    h_rigid_class.resize(atomic::CB_END);
    thrust::copy(sphcc, sphcc+atomic::CB_END, h_rigid_class.begin());
    h_rigid->classinfo = h_rigid_class;
}


struct _CopyFluid
{
    DeviceFluidDataSet  dfd;
    sphFluidParticle    *gl_partcle;
    float4              *gl_lights;

    _CopyFluid(DeviceFluidDataSet f, sphFluidParticle *glp, float4 *gll)
        : dfd(f), gl_partcle(glp), gl_lights(gll) {}

    __device__ void operator()(int i)
    {
        const uint P_ID = i;
        int pid = dfd.particles[P_ID].id;
        gl_partcle[P_ID] = dfd.particles[P_ID];

        int light_cycle = SPH_MAX_FLUID_PARTICLES/SPH_MAX_LIGHT_NUM;
        if(pid % light_cycle==0) {
            gl_lights[pid/light_cycle] = dfd.particles[P_ID].position;
        }
    }
};

void SPHCopyToGL()
{
    sphFluidParticle *gl_fluid = (sphFluidParticle*)h_fluid_gl.mapBuffer();
    sphRigidParticle *gl_rigid = (sphRigidParticle*)h_rigids_gl.mapBuffer();
    float4 *gl_lights = (float4*)h_light_gl.mapBuffer();

    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator((int)h_fluid->particles.size()),
        _CopyFluid(h_fluid->getDeviceData(), gl_fluid, gl_lights));

    //thrust::copy(d_rigid_p.begin(), d_rigid_p.end(), thrust::device_ptr<sphFluidParticle>(gl_fluid));
    thrust::copy(h_rigid->particles.begin(), h_rigid->particles.end(), thrust::device_ptr<sphRigidParticle>(gl_rigid));

    h_fluid_gl.unmapBuffer();
    h_rigids_gl.unmapBuffer();
    h_light_gl.unmapBuffer();
}



void SPHUpdateRigids(const thrust::host_vector<sphRigidInstance> &rigids)
{
    h_rigid->instances = rigids;

    int total = 0;
    for(uint ii=0; ii<rigids.size(); ++ii) {
        int classid = rigids[ii].classid;
        total += h_rigid_class[classid].num_particles;
    }
    h_rigid->particles.resize(total);
    h_states.num_rigid_particles = total;

    int n = 0;
    h_rigid_ui.resize(total);
    for(uint ii=0; ii<rigids.size(); ++ii) {
        int classid = rigids[ii].classid;
        sphRigidClass &cc = h_rigid_class[classid];
        for(uint pi=0; pi<cc.num_particles; ++pi) {
            h_rigid_ui[n+pi].cindex = ii;
            h_rigid_ui[n+pi].pindex = pi;
            h_rigid_ui[n+pi].classid = classid;
            h_rigid_ui[n+pi].owner_handle = rigids[ii].handle;
        }
        n += cc.num_particles;
    }
    h_rigid->updateinfo = h_rigid_ui;

    thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(total), _RigidUpdate(h_rigid->getDeviceData()) );
}


void SPHUpdateGravity(sphSphericalGravity (&sgravity)[ SPH_MAX_SPHERICAL_GRAVITY_NUM ])
{
    thrust::copy(sgravity, sgravity+SPH_MAX_SPHERICAL_GRAVITY_NUM, h_forces->sgravities.begin());
}


void SPHCopyDamageMessageToHost(sphDamageMessage *dst)
{
}



sphStates& SPHGetStates()
{
    return h_states;
}

