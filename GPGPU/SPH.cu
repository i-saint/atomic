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



__constant__ SPHParam d_params;
__device__ SPHCharacterClass d_cclass[atomic::CB_END];

struct SPHGravitySet
{
    SPHSphericalGravity *sgravity;
};

struct SPHFluidParticleSet
{
    SPHGridParam            *params;
    SPHFluidParticle        *particles;
    SPHFluidParticleForce   *forces;
    SPHHash                 *hashes;
    SPHGridData             *grid;
    SPHGPUStates            *states;

    __device__ int3 GridCalculateCell(float4 pos)
    {
        float4 c = (pos-params->grid_pos)*params->grid_dim_rcp;
        int3 uc = make_int3(c.x, c.y, c.z);
        return clamp(uc, make_int3(0), make_int3(SPH_GRID_DIV_X-1, SPH_GRID_DIV_Y-1, SPH_GRID_DIV_Z-1));
    }

    __device__ uint GridCalculateHash(float4 pos)
    {
        return GridConstuctKey( GridCalculateCell(pos) );
    }

    __device__ uint GridConstuctKey(int3 v)
    {
        return v.x | (v.y<<SPH_GRID_DIV_SHIFT_X) | (v.z<<(SPH_GRID_DIV_SHIFT_X+SPH_GRID_DIV_SHIFT_Y));
    }


    __device__ float CalculatePressure(float density)
    {
        // Implements this equation:
        // Pressure = B * ((rho / rho_0)^y  - 1)
        return d_params.pressure_stiffness * max(pow(density / d_params.rest_density, 3) - 1.0f, 0.0f);
    }

    __device__ float4 CalculateGradPressure(float r, float P_pressure, float N_pressure, float N_density, float4 diff)
    {
        const float h = d_params.smooth_len;
        float avg_pressure = 0.5f * (N_pressure + P_pressure);
        // Implements this equation:
        // W_spkiey(r, h) = 15 / (pi * h^6) * (h - r)^3
        // GRAD( W_spikey(r, h) ) = -45 / (pi * h^6) * (h - r)^2
        // g_fGradPressureCoef = fParticleMass * -45.0f / (PI * fSmoothlen^6)
        return (d_params.grad_pressure_coef * avg_pressure / N_density * (h - r) * (h - r) / r) * diff;
    }

    __device__ float4 CalculateLapVelocity(float r, float4 P_velocity, float4 N_velocity, float N_density)
    {
        const float h = d_params.smooth_len;
        float4 vel_diff = N_velocity - P_velocity;
        // Implements this equation:
        // W_viscosity(r, h) = 15 / (2 * pi * h^3) * (-r^3 / (2 * h^3) + r^2 / h^2 + h / (2 * r) - 1)
        // LAPLACIAN( W_viscosity(r, h) ) = 45 / (pi * h^6) * (h - r)
        // g_fLapViscosityCoef = fParticleMass * fViscosity * 45.0f / (PI * fSmoothlen^6)
        return  (d_params.lap_viscosity_coef / N_density * (h - r)) * vel_diff;
    }

    __device__ float CalculateDensity(float r_sq)
    {
        const float h_sq = d_params.smooth_len * d_params.smooth_len;
        // Implements this equation:
        // W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
        // g_fDensityCoef = fParticleMass * 315.0f / (64.0f * PI * fSmoothlen^9)
        return d_params.density_coef * (h_sq - r_sq) * (h_sq - r_sq) * (h_sq - r_sq);
    }
};

struct SPHRigidParticleSet
{
    SPHGridParam            *params;
    SPHRigidParticle        *particles;
    SPHHash                 *hashes;
    SPHGridData             *grid;
    SPHGPUStates            *states;

    __device__ int3 GridCalculateCell(float4 pos)
    {
        float4 c = (pos-params->grid_pos)*params->grid_dim_rcp;
        int3 uc = make_int3(c.x, c.y, c.z);
        return clamp(uc, make_int3(0), make_int3(SPH_GRID_DIV_X-1, SPH_GRID_DIV_Y-1, SPH_GRID_DIV_Z-1));
    }

    __device__ uint GridCalculateHash(float4 pos)
    {
        return GridConstuctKey( GridCalculateCell(pos) );
    }

    __device__ uint GridConstuctKey(int3 v)
    {
        return v.x | (v.y<<SPH_GRID_DIV_SHIFT_X) | (v.z<<(SPH_GRID_DIV_SHIFT_X+SPH_GRID_DIV_SHIFT_Y));
    }
};

struct SPHForceSet
{
    SPHSphericalGravity *sgravities;
};


SPHFluidParticleSet h_fluid;
SPHRigidParticleSet h_rigid;
SPHGravitySet h_gravities;



__device__ int GetThreadId()
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}


__global__ void GClearParticles(SPHFluidParticleSet ps)
{
    const float spacing = 0.009f;
    int i = GetThreadId();
    ps.particles[i].id = i;
    ps.particles[i].lifetime = 0xffffffff;
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
    CUDA_SAFE_CALL( cudaMalloc(&h_fluid.params,     sizeof(SPHGridParam)) );
    CUDA_SAFE_CALL( cudaMalloc(&h_fluid.particles,  sizeof(SPHFluidParticle)*SPH_MAX_FLUID_PARTICLES) );
    CUDA_SAFE_CALL( cudaMalloc(&h_fluid.forces,     sizeof(SPHFluidParticleForce)*SPH_MAX_FLUID_PARTICLES) );
    CUDA_SAFE_CALL( cudaMalloc(&h_fluid.hashes,     sizeof(SPHHash)*SPH_MAX_FLUID_PARTICLES) );
    CUDA_SAFE_CALL( cudaMalloc(&h_fluid.grid,       sizeof(SPHGridData)*SPH_GRID_DIV_3) );
    CUDA_SAFE_CALL( cudaMalloc(&h_fluid.states,     sizeof(SPHGPUStates)) );

    CUDA_SAFE_CALL( cudaMalloc(&h_rigid.params,     sizeof(SPHGridParam)) );
    CUDA_SAFE_CALL( cudaMalloc(&h_rigid.particles,  sizeof(SPHRigidParticle)*SPH_MAX_FLUID_PARTICLES) );
    CUDA_SAFE_CALL( cudaMalloc(&h_rigid.hashes,     sizeof(SPHHash)*SPH_MAX_FLUID_PARTICLES) );
    CUDA_SAFE_CALL( cudaMalloc(&h_rigid.grid,       sizeof(SPHGridData)*SPH_GRID_DIV_3) );
    CUDA_SAFE_CALL( cudaMalloc(&h_rigid.states,     sizeof(SPHGPUStates)) );

    CUDA_SAFE_CALL( cudaMalloc(&h_gravities.sgravity, sizeof(SPHSphericalGravity)*SPH_MAX_SPHERICAL_GRAVITY_NUM) );

    {
        SPHParam sph_params;
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

        SPHGridParam grid_params;
        const float grid_len = 5.12f;
        grid_params.grid_dim = make_float4(grid_len, grid_len, sph_params.smooth_len*SPH_GRID_DIV_Z, 0.0f);
        grid_params.grid_dim_rcp = make_float4(1.0f) / (grid_params.grid_dim / make_float4(SPH_GRID_DIV_X, SPH_GRID_DIV_Y, SPH_GRID_DIV_Z, 1.0));
        grid_params.grid_pos = make_float4(-grid_len/2.0f, -grid_len/2.0f, 0.0f, 0.0f);
        CUDA_SAFE_CALL( cudaMemcpy(h_fluid.params, &grid_params, sizeof(grid_params), cudaMemcpyHostToDevice) );
    }
    {
        SPHSphericalGravity h_sg;
        h_sg.position = make_float4(0.0f);
        h_sg.is_active = 1;
        h_sg.inner_radus = 0.5f;
        h_sg.range_radus = 5.12f;
        h_sg.strength = 0.5f;
        cudaMemcpy(h_gravities.sgravity, &h_sg, sizeof(h_sg), cudaMemcpyHostToDevice );
    }

    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_FLUID_PARTICLES / SPH_THREAD_BLOCK_X );
    GClearParticles<<<dimGrid, dimBlock>>>(h_fluid);
}

void SPHFinalize()
{
    CUDA_SAFE_CALL( cudaFree(h_gravities.sgravity) );

    CUDA_SAFE_CALL( cudaFree(h_rigid.states) );
    CUDA_SAFE_CALL( cudaFree(h_rigid.grid) );
    CUDA_SAFE_CALL( cudaFree(h_rigid.hashes) );
    CUDA_SAFE_CALL( cudaFree(h_rigid.particles) );
    CUDA_SAFE_CALL( cudaFree(h_rigid.params) );

    CUDA_SAFE_CALL( cudaFree(h_fluid.states) );
    CUDA_SAFE_CALL( cudaFree(h_fluid.grid) );
    CUDA_SAFE_CALL( cudaFree(h_fluid.hashes) );
    CUDA_SAFE_CALL( cudaFree(h_fluid.forces) );
    CUDA_SAFE_CALL( cudaFree(h_fluid.particles) );
    CUDA_SAFE_CALL( cudaFree(h_fluid.params) );
}



__global__ void GUpdateHash(SPHFluidParticleSet ps)
{
    const int i = GetThreadId();

    uint hash = ps.GridCalculateHash(ps.particles[i].position);
    ps.hashes[i] = hash;
}

__global__ void GZeroClearGrid(SPHFluidParticleSet ps)
{
    const int i = GetThreadId();

    ps.grid[i].x = ps.grid[i].y = 0;
}

__global__ void GUpdateGrid(SPHFluidParticleSet ps)
{
    const uint G_ID = GetThreadId();
    uint G_ID_PREV = (G_ID == 0)? SPH_MAX_FLUID_PARTICLES : G_ID; G_ID_PREV--;
    uint G_ID_NEXT = G_ID + 1; if (G_ID_NEXT == SPH_MAX_FLUID_PARTICLES) { G_ID_NEXT = 0; }
    
    uint cell = ps.hashes[G_ID];
    uint cell_prev = ps.hashes[G_ID_PREV];
    uint cell_next = ps.hashes[G_ID_NEXT];
    if (cell != cell_prev)
    {
        // I'm the start of a cell
        ps.grid[cell].x = G_ID;
    }
    if (cell != cell_next)
    {
        // I'm the end of a cell
        ps.grid[cell].y = G_ID + 1;
    }
}


void SPHUpdateGrid()
{
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid_par_particle( SPH_MAX_FLUID_PARTICLES / SPH_THREAD_BLOCK_X );
    dim3 dimGrid_par_grid( SPH_GRID_DIV_3 / SPH_THREAD_BLOCK_X );

    GUpdateHash<<<dimGrid_par_particle, dimBlock>>>(h_fluid);

    // thrust::sort_by_key 用にデバイス側のポインタを取得
    // *直接 thrust::sort_by_key(d_hashes, d_hashes+SPH_MAX_FLUID_PARTICLES, d_particles) とかやると、
    //  コンパイルエラーにはならないけど意図した結果にならない (host 側用の関数が呼ばれる)
    thrust::device_ptr<SPHHash> dphashes(h_fluid.hashes);
    thrust::device_ptr<SPHFluidParticle> dpparticles(h_fluid.particles);

    thrust::sort_by_key(dphashes, dphashes+SPH_MAX_FLUID_PARTICLES, dpparticles);
    GZeroClearGrid<<<dimGrid_par_grid, dimBlock>>>(h_fluid);
    GUpdateGrid<<<dimGrid_par_particle, dimBlock>>>(h_fluid);
}



__global__ void GComputeDensity(SPHFluidParticleSet ps)
{
    const uint P_ID = GetThreadId();
    const float h_sq = d_params.smooth_len * d_params.smooth_len;
    float4 P_position = ps.particles[P_ID].position;

    float density = 0.0f;

    int3 G_XYZ = ps.GridCalculateCell( P_position );
    for(int Z = max(G_XYZ.z - 1, 0) ; Z <= min(G_XYZ.z + 1, SPH_GRID_DIV_Z-1) ; Z++)
    {
        for(int Y = max(G_XYZ.y - 1, 0) ; Y <= min(G_XYZ.y + 1, SPH_GRID_DIV_Y-1) ; Y++)
        {
            for(int X = max(G_XYZ.x - 1, 0) ; X <= min(G_XYZ.x + 1, SPH_GRID_DIV_X-1) ; X++)
            {
                SPHHash G_CELL = ps.GridConstuctKey(make_int3(X, Y, Z));
                SPHGridData G_START_END = ps.grid[G_CELL];
                for(uint N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
                {
                    float4 N_position = ps.particles[N_ID].position;
                
                    float4 diff = N_position - P_position;
                    float r_sq = dot(diff, diff);
                    if(r_sq < h_sq)
                    {
                        density += ps.CalculateDensity(r_sq);
                    }
                }
            }
        }
    }

    ps.forces[P_ID].density = density;
}

void SPHComputeDensity()
{
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_FLUID_PARTICLES / SPH_THREAD_BLOCK_X );

    GComputeDensity<<<dimGrid, dimBlock>>>(h_fluid);
}




__global__ void GComputeForce(SPHFluidParticleSet ps)
{
    const uint P_ID = GetThreadId();
    
    float4 P_position = ps.particles[P_ID].position;
    float4 P_velocity = ps.particles[P_ID].velocity;
    float P_density = ps.forces[P_ID].density;
    float P_pressure = ps.CalculatePressure(P_density);
    
    const float h_sq = d_params.smooth_len * d_params.smooth_len;
    
    float4 acceleration = make_float4(0);

    // Calculate the acceleration based on all neighbors
    int3 G_XYZ = ps.GridCalculateCell( P_position );
    for(int Z = max(G_XYZ.z - 1, 0) ; Z <= min(G_XYZ.z + 1, SPH_GRID_DIV_Z-1) ; Z++)
    {
        for(int Y = max(G_XYZ.y - 1, 0) ; Y <= min(G_XYZ.y + 1, SPH_GRID_DIV_Y-1) ; Y++)
        {
            for(int X = max(G_XYZ.x - 1, 0) ; X <= min(G_XYZ.x + 1, SPH_GRID_DIV_X-1) ; X++)
            {
                SPHHash G_CELL = ps.GridConstuctKey(make_int3(X, Y, Z));
                SPHGridData G_START_END = ps.grid[G_CELL];
                for(uint N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
                {
                    float4 N_position = ps.particles[N_ID].position;

                    float4 diff = N_position - P_position;
                    float r_sq = dot(diff, diff);
                    if(r_sq < h_sq && P_ID != N_ID)
                    {
                        float4 N_velocity = ps.particles[N_ID].velocity;
                        float N_density = ps.forces[N_ID].density;
                        float N_pressure = ps.CalculatePressure(N_density);
                        float r = sqrt(r_sq);

                        // Pressure Term
                        acceleration += ps.CalculateGradPressure(r, P_pressure, N_pressure, N_density, diff);
            
                        // Viscosity Term
                        acceleration += ps.CalculateLapVelocity(r, P_velocity, N_velocity, N_density);
                    }
                }
            }
        }
    }

    ps.forces[P_ID].acceleration = acceleration / P_density;
}

void SPHComputeForce()
{
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_FLUID_PARTICLES / SPH_THREAD_BLOCK_X );

    GComputeForce<<<dimGrid, dimBlock>>>(h_fluid);
}



__global__ void GIntegrate(SPHFluidParticleSet ps, SPHGravitySet gs)
{
    const uint P_ID = GetThreadId();

    float4 position = ps.particles[P_ID].position;
    float4 velocity = ps.particles[P_ID].velocity;
    float4 acceleration = ps.forces[P_ID].acceleration;

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
        if(!gs.sgravity[i].is_active) { continue; }

        const float4 center = gs.sgravity[i].position;
        const float gravity_strength = gs.sgravity[i].strength;
        const float inner_radius = gs.sgravity[i].inner_radus;
        const float outer_radius = gs.sgravity[i].range_radus;

        float4 diff = center-position;
        diff.w = 0.0f;
        float distance = length(diff);
        float4 dir = diff/distance;
        float4 gravity = dir * gravity_strength;

        acceleration += min(distance-inner_radius, 0.0f) * d_params.wall_stiffness * dir;
        acceleration += min(outer_radius-distance, 0.0f) * -d_params.wall_stiffness * dir;
        acceleration += gravity;
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
    ps.particles[P_ID].density = ps.forces[P_ID].density;
    ps.particles[P_ID].position = position;
    ps.particles[P_ID].velocity = velocity;
}

void SPHIntegrate()
{
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_FLUID_PARTICLES / SPH_THREAD_BLOCK_X );

    GIntegrate<<<dimGrid, dimBlock>>>(h_fluid, h_gravities);
}

void SPHUpdateFluid()
{
    SPHUpdateGrid();
    SPHComputeDensity();
    SPHComputeForce();
    SPHIntegrate();
}


__global__ void GCopyInstances(SPHFluidParticle *d_fractions, float4 *d_lights, SPHFluidParticleSet ps)
{
    const uint P_ID = GetThreadId();
    int pid = ps.particles[P_ID].id;
    d_fractions[P_ID] = ps.particles[P_ID];

    int light_cycle = SPH_MAX_FLUID_PARTICLES/SPH_MAX_LIGHT_NUM;
    if(pid % light_cycle==0) {
        d_lights[pid/light_cycle] = ps.particles[P_ID].position;
    }
}




DeviceBufferObject h_fluid_gl;
DeviceBufferObject h_rigids_gl;
DeviceBufferObject h_light_gl;
SPHCharacterClass h_sphcc[atomic::CB_END];

void SPHInitializeInstanceBuffers(int vbo_fluid, int vbo_rigids, int vbo_lightpos)
{
    h_fluid_gl.registerBuffer(vbo_fluid, cudaGraphicsMapFlagsWriteDiscard);
    h_rigids_gl.registerBuffer(vbo_rigids, cudaGraphicsMapFlagsWriteDiscard);
    h_light_gl.registerBuffer(vbo_lightpos, cudaGraphicsMapFlagsWriteDiscard);
}

void SPHFinalizeInstanceBuffers()
{
    h_light_gl.unregisterBuffer();
    h_rigids_gl.unregisterBuffer();
    h_fluid_gl.unregisterBuffer();
}

void SPHCopyClassInfo(SPHCharacterClass (&sphcc)[atomic::CB_END])
{
    thrust::copy(sphcc, sphcc+atomic::CB_END, h_sphcc);
    cudaMemcpyToSymbol("d_cclass", sphcc, sizeof(sphcc));
}

void SPHCopyToGL()
{
    SPHFluidParticle *d_fluid = (SPHFluidParticle*)h_fluid_gl.mapBuffer();
    SPHRigidParticle *d_rigid = (SPHRigidParticle*)h_rigids_gl.mapBuffer();
    float4 *d_lights = (float4*)h_light_gl.mapBuffer();

    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_FLUID_PARTICLES / SPH_THREAD_BLOCK_X );
    GCopyInstances<<<dimGrid, dimBlock>>>(d_fluid, d_lights, h_fluid);

    h_fluid_gl.unmapBuffer();
    h_rigids_gl.unmapBuffer();
    h_light_gl.unmapBuffer();
}

struct SPHRigidUpdateInfo
{
    int cindex;
    int pindex;
    int classid;
    EntityHandle owner_handle;
};

template<class T, class U>
__device__ __host__ T& vector_cast(U& v) { return reinterpret_cast<T&>(v); }

struct SPHRigidUpdater
{
    SPHCharacterClass       *sphcc;
    SPHCharacterInstance    *sphci;

    template <typename Tuple>
    __device__ void operator()(Tuple t)
    {
        SPHRigidUpdateInfo      &rui = thrust::get<0>(t);
        SPHRigidParticle        &rp = thrust::get<1>(t);
        SPHCharacterClass       &cc = sphcc[rui.classid];
        SPHCharacterInstance    &ins = sphci[rui.cindex];
        rp.owner_handle = rui.owner_handle;
        rp.position     = vector_cast<float4&>(ins.transform * vector_cast<vec4>(cc.particles[rui.pindex].position));
        rp.normal       = vector_cast<float4&>(ins.transform * vector_cast<vec4>(cc.particles[rui.pindex].normal));
    }
};

void SPHUpdateRigids(const thrust::host_vector<SPHCharacterInstance> &rigids)
{
    thrust::device_vector<SPHCharacterInstance> d_instances;
    thrust::device_vector<SPHRigidParticle>     d_rigid;
    thrust::device_vector<SPHRigidUpdateInfo>   d_rui;
    thrust::host_vector<SPHRigidUpdateInfo>     h_rui;

    d_instances.resize(rigids.size());
    thrust::copy(rigids.begin(), rigids.end(), d_instances.begin());

    uint total = 0;
    for(uint ii=0; ii<rigids.size(); ++ii) {
        int classid = rigids[ii].classid;
        total += h_sphcc[classid].num_particles;
    }
    d_rigid.resize(total);
    d_rui.resize(total);
    h_rui.resize(total);

    uint n = 0;
    for(uint ii=0; ii<rigids.size(); ++ii) {
        int classid = rigids[ii].classid;
        SPHCharacterClass &cc = h_sphcc[classid];
        for(uint pi=0; pi<cc.num_particles; ++pi) {
            h_rui[n+pi].cindex = ii;
            h_rui[n+pi].pindex = pi;
            h_rui[n+pi].classid = classid;
            h_rui[n+pi].owner_handle = rigids[ii].handle;
        }
        n += cc.num_particles;
    }

    SPHRigidUpdater updator;
    updator.sphcc = h_sphcc;
    updator.sphci = d_instances.data().get();
    thrust::copy(h_rui.begin(), h_rui.end(), d_rui.begin());
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_rui.begin(), d_rigid.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(d_rui.end(),   d_rigid.end()  )),
                     updator);
}


void SPHUpdateGravity(SPHSphericalGravity (&sgravity)[ SPH_MAX_SPHERICAL_GRAVITY_NUM ])
{
    cudaMemcpy(h_gravities.sgravity, sgravity, sizeof(sgravity), cudaMemcpyHostToDevice );
}


void SPHCopyDamageMessageToHost(SPHDamageMessage *dst)
{
}


void SPHSpawnFluidParticles(const thrust::host_vector<SPHCharacterInstance> &rigids)
{
}

