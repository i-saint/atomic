#ifndef __atomic_SPH_cuh__
#define __atomic_SPH_cuh__

extern "C" {

    const int SPH_GRID_DIV_SHIFT_X = 9; // 
    const int SPH_GRID_DIV_SHIFT_Y = 9; // 
    const int SPH_GRID_DIV_SHIFT_Z = 3; // 
    const int SPH_GRID_DIV_X = 1<<SPH_GRID_DIV_SHIFT_X;
    const int SPH_GRID_DIV_Y = 1<<SPH_GRID_DIV_SHIFT_Y;
    const int SPH_GRID_DIV_Z = 1<<SPH_GRID_DIV_SHIFT_Z;
    const int SPH_GRID_DIV_3 = SPH_GRID_DIV_X*SPH_GRID_DIV_Y*SPH_GRID_DIV_Z;
    //const int SPH_MAX_PARTICLE_NUM = 65536*2;
    const int SPH_MAX_PARTICLE_NUM = 65536;
    const int SPH_MAX_LIGHT_NUM = 16;
    const int SPH_MAX_SPHERICAL_GRAVITY_NUM = 1;
    const int SPH_THREAD_BLOCK_X = 256;


    struct SPHGPUStates
    {
        int num_particles;
    };

    struct SPHParticle
    {
        union {
            struct {
                int id;
                int lifetime;
                float density;
            };
            float4 padding;
        };
        float4 position;
        float4 velocity;
    };

    struct SPHParticleForce
    {
        float4 acceleration;
        union {
            struct {
                float density;
            };
            float4 padding;
        };
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


    struct SPHCharacterClassInfo
    {
        union {
            struct {
                int classid;
                int num_particles;
                float4 *particles;
            };
            float4 padding;
        };
    };

    void SPHInitialize();
    void SPHFinalize();
    void SPHUpdate();

    void SPHInitializeInstanceBuffers(int vbo_fraction, int vbo_lightpos);
    void SPHFinalizeInstanceBuffers();
    void SPHCopyInstances();

    void SPHUpdateSphericalGravityData(SPHSphericalGravity (&sgravity)[ SPH_MAX_SPHERICAL_GRAVITY_NUM ]);

    void SPHSpawnParticles(const SPHParticle* spawn, int num_pawn);
    void SPHSpawnSolidParticles(const SPHParticle* spawn, int num_pawn);

} // extern "C"

#endif // __atomic_SPH_cuh__
