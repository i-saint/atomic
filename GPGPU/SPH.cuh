extern "C" {

    const int SPH_GRID_DIV_SHIFT_X = 8; // 256
    const int SPH_GRID_DIV_SHIFT_Y = 8; // 256
    const int SPH_GRID_DIV_SHIFT_Z = 0; // 1
    const int SPH_GRID_DIV_X = 1<<SPH_GRID_DIV_SHIFT_X;
    const int SPH_GRID_DIV_Y = 1<<SPH_GRID_DIV_SHIFT_Y;
    const int SPH_GRID_DIV_Z = 1<<SPH_GRID_DIV_SHIFT_Z;
    const int SPH_GRID_DIV_3 = SPH_GRID_DIV_X*SPH_GRID_DIV_Y*SPH_GRID_DIV_Z;
    const int SPH_MAX_PARTICLE_NUM = 32768;
    //const int SPH_MAX_PARTICLE_NUM = 4096;

    struct SPHParam
    {
        float smooth_len;
        float pressure_stiffness;
        float rest_density;
        float particle_mass;
        float viscosity;
        float density_coef;
        float grad_pressure_coef;
        float lap_viscosity_coef;
        float wall_stiffness;
        float4 grid_dim;
        float4 grid_dim_rcp;
        float4 grid_pos;
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

    void SPHInitialize();
    void SPHUpdateGrid();
    void SPHComputeDensity();
    void SPHComputeForce();
    void SPHIntegrate();

} // extern "C"
