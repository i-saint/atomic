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
    //const int SPH_MAX_PARTICLE_NUM = 262144;
    const int SPH_MAX_PARTICLE_NUM = 65536;
    const int SPH_MAX_LIGHT_NUM = 16;
    const int SPH_MAX_SPHERICAL_GRAVITY_NUM = 1;

    struct SPHParam
    {
        float4 grid_dim;
        float4 grid_dim_rcp;
        float4 grid_pos;
        float smooth_len;
        float pressure_stiffness;
        float rest_density;
        float particle_mass;
        float viscosity;
        float density_coef;
        float grad_pressure_coef;
        float lap_viscosity_coef;
        float wall_stiffness;
    };

    struct SPHParticle
    {
        union {
            struct {
                int id;
                int lifetime;
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

    void SPHInitialize();
    void SPHFinalize();

    void SPHUpdateGrid();
    void SPHComputeDensity();
    void SPHComputeForce();
    void SPHIntegrate();

    void SPHInitializeInstancePositionBuffer(int vbo_inspos, int vbo_lightpos);
    void SPHFinalizeInstancePositionBuffer();
    void SPHCopyInstancePositions();

    void SPHUpdateSphericalGravityData(SPHSphericalGravity (&sgravity)[ SPH_MAX_SPHERICAL_GRAVITY_NUM ]);

} // extern "C"

#endif // __atomic_SPH_cuh__
