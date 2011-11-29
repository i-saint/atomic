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

typedef unsigned int uint;


#define SPH_ENABLE_HASHGRID



__constant__ __align__(16) SPHParam       d_param;
__device__ __align__(16) SPHParticle      d_particles[ SPH_MAX_PARTICLE_NUM ];
__device__ __align__(16) SPHParticleForce d_forces[ SPH_MAX_PARTICLE_NUM ];
__device__ __align__(16) uint             d_hashes[ SPH_MAX_PARTICLE_NUM ];
__device__ __align__(16) uint2            d_grid[ SPH_GRID_DIV_3 ];

__device__ __align__(16) SPHSphericalGravity d_sgravity[ SPH_MAX_SPHERICAL_GRAVITY_NUM ];


const int SPH_THREAD_BLOCK_X = 256;


__device__ int GetThreadId()
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}


__global__ void GClearParticles()
{
    const float spacing = 0.009f;
    int i = GetThreadId();
    d_particles[i].id = i;
    d_particles[i].lifetime = 0xffffffff;
    uint w = 128;
    d_particles[i].position = make_float4(
        spacing*(i%w) - (spacing*w*0.5),
        spacing*((i/w)%w) + 0.6,
        /*0.0f,*/ spacing*(i/(w*w))+0.05f,
        0.0f);
    d_particles[i].velocity = make_float4(0.0f);

    d_forces[i].density = 0.0f;
    d_forces[i].acceleration = make_float4(0.0f);
}

void SPHInitialize()
{
    {
        SPHParam h_param;
        h_param.smooth_len          = 0.02f;
        h_param.pressure_stiffness  = 200.0f;
        h_param.rest_density        = 1000.0f;
        h_param.particle_mass       = 0.001f;
        h_param.viscosity           = 0.1f;
        h_param.density_coef        = h_param.particle_mass * 315.0f / (64.0f * CUDART_PI_F * pow(h_param.smooth_len, 9));
        h_param.grad_pressure_coef  = h_param.particle_mass * -45.0f / (CUDART_PI_F * pow(h_param.smooth_len, 6));
        h_param.lap_viscosity_coef  = h_param.particle_mass * h_param.viscosity * 45.0f / (CUDART_PI_F * pow(h_param.smooth_len, 6));
        h_param.wall_stiffness      = 3000.0f;
        const float grid_len = 5.12f;
        h_param.grid_dim = make_float4(grid_len, grid_len, h_param.smooth_len*SPH_GRID_DIV_Z, 0.0f);
        h_param.grid_dim_rcp = make_float4(1.0f) / (h_param.grid_dim / make_float4(SPH_GRID_DIV_X, SPH_GRID_DIV_Y, SPH_GRID_DIV_Z, 1.0));
        h_param.grid_pos = make_float4(-grid_len/2.0f, -grid_len/2.0f, 0.0f, 0.0f);
        CUDA_SAFE_CALL( cudaMemcpyToSymbol("d_param", &h_param, sizeof(h_param)) );
    }
    {
        SPHSphericalGravity h_sg;
        h_sg.position = make_float4(0.0f);
        h_sg.is_active = 1;
        h_sg.inner_radus = 0.5f;
        h_sg.range_radus = 5.12f;
        h_sg.strength = 0.5f;
        CUDA_SAFE_CALL( cudaMemcpyToSymbol("d_sgravity", &h_sg, sizeof(h_sg)) );
    }

    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_PARTICLE_NUM / SPH_THREAD_BLOCK_X );
    GClearParticles<<<dimGrid, dimBlock>>>();
}

void SPHFinalize()
{

}


__device__ int3 GridCalculateCell(float4 pos)
{
    float4 c = (pos-d_param.grid_pos)*d_param.grid_dim_rcp;
    int3 uc = make_int3(c.x, c.y, c.z);
    return clamp(uc, make_int3(0), make_int3(SPH_GRID_DIV_X-1, SPH_GRID_DIV_Y-1, SPH_GRID_DIV_Z-1));
}

__device__ uint GridConstuctKey(int3 v)
{
    return v.x | (v.y<<SPH_GRID_DIV_SHIFT_X) | (v.z<<(SPH_GRID_DIV_SHIFT_X+SPH_GRID_DIV_SHIFT_Y));
}

__device__ uint GridCalculateHash(float4 pos)
{
    return GridConstuctKey( GridCalculateCell(pos) );
}


__global__ void GUpdateHash()
{
    const int i = GetThreadId();

    uint hash = GridCalculateHash(d_particles[i].position);
    d_hashes[i] = hash;
}

__global__ void GZeroClearGrid()
{
    const int i = GetThreadId();

    d_grid[i].x = d_grid[i].y = 0;
}

__global__ void GUpdateGrid()
{
    const unsigned int G_ID = GetThreadId();
    unsigned int G_ID_PREV = (G_ID == 0)? SPH_MAX_PARTICLE_NUM : G_ID; G_ID_PREV--;
    unsigned int G_ID_NEXT = G_ID + 1; if (G_ID_NEXT == SPH_MAX_PARTICLE_NUM) { G_ID_NEXT = 0; }
    
    unsigned int cell = d_hashes[G_ID];
    unsigned int cell_prev = d_hashes[G_ID_PREV];
    unsigned int cell_next = d_hashes[G_ID_NEXT];
    if (cell != cell_prev)
    {
        // I'm the start of a cell
        d_grid[cell].x = G_ID;
    }
    if (cell != cell_next)
    {
        // I'm the end of a cell
        d_grid[cell].y = G_ID + 1;
    }
}


void SPHUpdateGrid()
{
#ifdef SPH_ENABLE_HASHGRID
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid_par_particle( SPH_MAX_PARTICLE_NUM / SPH_THREAD_BLOCK_X );
    dim3 dimGrid_par_grid( SPH_GRID_DIV_3 / SPH_THREAD_BLOCK_X );

    GUpdateHash<<<dimGrid_par_particle, dimBlock>>>();

    // thrust::sort_by_key 用にデバイス側のポインタを取得
    // *直接 thrust::sort_by_key(d_hashes, d_hashes+SPH_MAX_PARTICLE_NUM, d_particles) とかやると、
    //  コンパイルエラーにはならないけど意図した結果にならない (host 側用の関数が呼ばれる)
    uint *phashes = NULL;
    SPHParticle *pparticles = NULL;
    CUDA_SAFE_CALL( cudaGetSymbolAddress((void**)&phashes, "d_hashes") );
    CUDA_SAFE_CALL( cudaGetSymbolAddress((void**)&pparticles, "d_particles") );
    thrust::device_ptr<uint> dphashes(phashes);
    thrust::device_ptr<SPHParticle> dpparticles(pparticles);

    thrust::sort_by_key(dphashes, dphashes+SPH_MAX_PARTICLE_NUM, dpparticles);
    GZeroClearGrid<<<dimGrid_par_grid, dimBlock>>>();
    GUpdateGrid<<<dimGrid_par_particle, dimBlock>>>();
#endif // SPH_ENABLE_HASHGRID
}



__device__ float CalculateDensity(float r_sq)
{
    const float h_sq = d_param.smooth_len * d_param.smooth_len;
    // Implements this equation:
    // W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
    // g_fDensityCoef = fParticleMass * 315.0f / (64.0f * PI * fSmoothlen^9)
    return d_param.density_coef * (h_sq - r_sq) * (h_sq - r_sq) * (h_sq - r_sq);
}

__global__ void GComputeDensity()
{
    const unsigned int P_ID = GetThreadId();
    const float h_sq = d_param.smooth_len * d_param.smooth_len;
    float4 P_position = d_particles[P_ID].position;

    float density = 0.0f;

#ifdef SPH_ENABLE_HASHGRID

    int3 G_XYZ = GridCalculateCell( P_position );
    for (int Z = max(G_XYZ.z - 1, 0) ; Z <= min(G_XYZ.z + 1, SPH_GRID_DIV_Z-1) ; Z++)
    {
        for (int Y = max(G_XYZ.y - 1, 0) ; Y <= min(G_XYZ.y + 1, SPH_GRID_DIV_Y-1) ; Y++)
        {
            for (int X = max(G_XYZ.x - 1, 0) ; X <= min(G_XYZ.x + 1, SPH_GRID_DIV_X-1) ; X++)
            {
                uint G_CELL = GridConstuctKey(make_int3(X, Y, Z));
                uint2 G_START_END = d_grid[G_CELL];
                for (unsigned int N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
                {
                    float4 N_position = d_particles[N_ID].position;
                
                    float4 diff = N_position - P_position;
                    float r_sq = dot(diff, diff);
                    if (r_sq < h_sq)
                    {
                        density += CalculateDensity(r_sq);
                    }
                }
            }
        }
    }

#else

    // Calculate the density based on all neighbors
    for(uint N_ID = 0 ; N_ID < SPH_MAX_PARTICLE_NUM ; N_ID++)
    {
        float4 N_position = d_particles[N_ID].position;
        
        float4 diff = N_position - P_position;
        float r_sq = dot(diff, diff);
        if (r_sq < h_sq)
        {
            density += CalculateDensity(r_sq);
        }
    }

#endif

    d_forces[P_ID].density = density;
}

void SPHComputeDensity()
{
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_PARTICLE_NUM / SPH_THREAD_BLOCK_X );

    GComputeDensity<<<dimGrid, dimBlock>>>();
}



__device__ float CalculatePressure(float density)
{
    // Implements this equation:
    // Pressure = B * ((rho / rho_0)^y  - 1)
    return d_param.pressure_stiffness * max(pow(density / d_param.rest_density, 3) - 1.0f, 0.0f);
}

__device__ float4 CalculateGradPressure(float r, float P_pressure, float N_pressure, float N_density, float4 diff)
{
    const float h = d_param.smooth_len;
    float avg_pressure = 0.5f * (N_pressure + P_pressure);
    // Implements this equation:
    // W_spkiey(r, h) = 15 / (pi * h^6) * (h - r)^3
    // GRAD( W_spikey(r, h) ) = -45 / (pi * h^6) * (h - r)^2
    // g_fGradPressureCoef = fParticleMass * -45.0f / (PI * fSmoothlen^6)
    return (d_param.grad_pressure_coef * avg_pressure / N_density * (h - r) * (h - r) / r) * diff;
}

__device__ float4 CalculateLapVelocity(float r, float4 P_velocity, float4 N_velocity, float N_density)
{
    const float h = d_param.smooth_len;
    float4 vel_diff = N_velocity - P_velocity;
    // Implements this equation:
    // W_viscosity(r, h) = 15 / (2 * pi * h^3) * (-r^3 / (2 * h^3) + r^2 / h^2 + h / (2 * r) - 1)
    // LAPLACIAN( W_viscosity(r, h) ) = 45 / (pi * h^6) * (h - r)
    // g_fLapViscosityCoef = fParticleMass * fViscosity * 45.0f / (PI * fSmoothlen^6)
    return  (d_param.lap_viscosity_coef / N_density * (h - r)) * vel_diff;
}

__global__ void GComputeForce()
{
    const unsigned int P_ID = GetThreadId();
    
    float4 P_position = d_particles[P_ID].position;
    float4 P_velocity = d_particles[P_ID].velocity;
    float P_density = d_forces[P_ID].density;
    float P_pressure = CalculatePressure(P_density);
    
    const float h_sq = d_param.smooth_len * d_param.smooth_len;
    
    float4 acceleration = make_float4(0);

#ifdef SPH_ENABLE_HASHGRID
    // Calculate the acceleration based on all neighbors
    int3 G_XYZ = GridCalculateCell( P_position );
    for (int Z = max(G_XYZ.z - 1, 0) ; Z <= min(G_XYZ.z + 1, SPH_GRID_DIV_Z-1) ; Z++)
    {
        for (int Y = max(G_XYZ.y - 1, 0) ; Y <= min(G_XYZ.y + 1, SPH_GRID_DIV_Y-1) ; Y++)
        {
            for (int X = max(G_XYZ.x - 1, 0) ; X <= min(G_XYZ.x + 1, SPH_GRID_DIV_X-1) ; X++)
            {
                uint G_CELL = GridConstuctKey(make_int3(X, Y, Z));
                uint2 G_START_END = d_grid[G_CELL];
                for (unsigned int N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
                {
                    float4 N_position = d_particles[N_ID].position;

                    float4 diff = N_position - P_position;
                    float r_sq = dot(diff, diff);
                    if (r_sq < h_sq && P_ID != N_ID)
                    {
                        float4 N_velocity = d_particles[N_ID].velocity;
                        float N_density = d_forces[N_ID].density;
                        float N_pressure = CalculatePressure(N_density);
                        float r = sqrt(r_sq);

                        // Pressure Term
                        acceleration += CalculateGradPressure(r, P_pressure, N_pressure, N_density, diff);
            
                        // Viscosity Term
                        acceleration += CalculateLapVelocity(r, P_velocity, N_velocity, N_density);
                    }
                }
            }
        }
    }

#else // SPH_ENABLE_HASHGRID

    // Calculate the acceleration based on all neighbors
    for (uint N_ID = 0 ; N_ID < SPH_MAX_PARTICLE_NUM ; N_ID++)
    {
        float4 N_position = d_particles[N_ID].position;
        
        float4 diff = N_position - P_position;
        float r_sq = dot(diff, diff);
        if (r_sq < h_sq && P_ID != N_ID)
        {
            float4 N_velocity = d_particles[N_ID].velocity;
            float N_density = d_forces[N_ID].density;
            float N_pressure = CalculatePressure(N_density);
            float r = sqrt(r_sq);

            // Pressure Term
            acceleration += CalculateGradPressure(r, P_pressure, N_pressure, N_density, diff);
            
            // Viscosity Term
            acceleration += CalculateLapVelocity(r, P_velocity, N_velocity, N_density);
        }
    }

#endif // SPH_ENABLE_HASHGRID

    d_forces[P_ID].acceleration = acceleration / P_density;
}

void SPHComputeForce()
{
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_PARTICLE_NUM / SPH_THREAD_BLOCK_X );

    GComputeForce<<<dimGrid, dimBlock>>>();
}



__global__ void GIntegrate()
{
    const unsigned int P_ID = GetThreadId();

    float4 position = d_particles[P_ID].position;
    float4 velocity = d_particles[P_ID].velocity;
    float4 acceleration = d_forces[P_ID].acceleration;

    //const float3 planes[4] = {
    //    make_float3( 1.0f, 0.0f, 0),
    //    make_float3( 0.0f, 1.0f, 0),
    //    make_float3(-1.0f, 0.0f, 2.56f),
    //    make_float3( 0.0f,-1.0f, 2.56f),
    //};
    //// Apply the forces from the map walls
    //for(unsigned int i = 0 ; i < 4 ; i++)
    //{
    //    float dist = dot(make_float3(position.x, position.y, 1.0f), planes[i]);
    //    acceleration += min(dist, 0.0f) * -d_param.wall_stiffness * make_float4(planes[i].x, planes[i].y, 0.0f, 0.0f);
    //}
    //float4 gravity = make_float4(0.0f, -0.5f, 0.0f, 0.0f);

    acceleration += min(position.z, 0.0f) * -d_param.wall_stiffness * make_float4(0.0f, 0.0f, 0.5f, 0.0f);
    acceleration += make_float4(0.0f, 0.0f, -5.0f, 0.0f);


    // Apply gravity
    for(int i=0; i<SPH_MAX_SPHERICAL_GRAVITY_NUM; ++i) {
        if(!d_sgravity[i].is_active) { continue; }

        const float4 center = d_sgravity[i].position;
        const float gravity_strength = d_sgravity[i].strength;
        const float inner_radius = d_sgravity[i].inner_radus;
        const float outer_radius = d_sgravity[i].range_radus;

        float4 diff = center-position;
        diff.w = 0.0f;
        float distance = length(diff);
        float4 dir = diff/distance;
        float4 gravity = dir * gravity_strength;

        acceleration += min(distance-inner_radius, 0.0f) * d_param.wall_stiffness * dir;
        acceleration += min(outer_radius-distance, 0.0f) * -d_param.wall_stiffness * dir;

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
    d_particles[P_ID].position = position;
    d_particles[P_ID].velocity = velocity;
}

void SPHIntegrate()
{
    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_PARTICLE_NUM / SPH_THREAD_BLOCK_X );

    GIntegrate<<<dimGrid, dimBlock>>>();
}


__global__ void GCopyInstancePositions(float4 *d_instance_pos, float4 *d_light_pos)
{
    const unsigned int P_ID = GetThreadId();
    int pid = d_particles[P_ID].id;
    d_instance_pos[P_ID] = d_particles[P_ID].position;

    int light_cycle = SPH_MAX_PARTICLE_NUM/SPH_MAX_LIGHT_NUM;
    if(pid % light_cycle==0) {
        d_light_pos[pid/light_cycle] = d_particles[P_ID].position;
    }
}




DeviceBufferObject h_instance_pos;
DeviceBufferObject h_light_pos;

void SPHInitializeInstancePositionBuffer(int vbo_inspos, int vbo_lightpos)
{
    h_instance_pos.registerBuffer(vbo_inspos, cudaGraphicsMapFlagsWriteDiscard);
    h_light_pos.registerBuffer(vbo_lightpos, cudaGraphicsMapFlagsWriteDiscard);
}

void SPHFinalizeInstancePositionBuffer()
{
    h_instance_pos.unregisterBuffer();
    h_light_pos.unregisterBuffer();
}

void SPHCopyInstancePositions()
{
    float4 *d_inspos = (float4*)h_instance_pos.mapBuffer();
    float4 *d_lightpos = (float4*)h_light_pos.mapBuffer();

    dim3 dimBlock( SPH_THREAD_BLOCK_X );
    dim3 dimGrid( SPH_MAX_PARTICLE_NUM / SPH_THREAD_BLOCK_X );
    GCopyInstancePositions<<<dimGrid, dimBlock>>>(d_inspos, d_lightpos);

    h_instance_pos.unmapBuffer();
    h_light_pos.unmapBuffer();
}


void SPHUpdateSphericalGravityData(SPHSphericalGravity (&sgravity)[ SPH_MAX_SPHERICAL_GRAVITY_NUM ])
{
    CUDA_SAFE_CALL( cudaMemcpyToSymbol("d_sgravity", sgravity, sizeof(sgravity)) );
}
