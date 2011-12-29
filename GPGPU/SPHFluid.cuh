
struct DeviceFluidDataSet
{
    sphGridParam        *params;
    sphFluidParticle    *particles;
    sphFluidForce       *forces;
    sphHash             *hashes;
    sphDeadFlag         *dead;
    sphFluidMessage     *message;
    sphGridData         *grid;
    sphStates           *states;

    __device__ int3 GridCalculateCell(float4 pos)
    {
        float4 c = (pos-params->grid_pos)*params->grid_dim_rcp;
        int3 uc = make_int3(c.x, c.y, c.z);
        return clamp(uc, make_int3(0), make_int3(SPH_FLUID_GRID_DIV_X-1, SPH_FLUID_GRID_DIV_Y-1, SPH_FLUID_GRID_DIV_Z-1));
    }

    __device__ uint GridCalculateHash(float4 pos)
    {
        return GridConstuctKey( GridCalculateCell(pos) );
    }

    __device__ uint GridConstuctKey(int3 v)
    {
        return v.x | (v.y<<SPH_FLUID_GRID_DIV_SHIFT_X) | (v.z<<(SPH_FLUID_GRID_DIV_SHIFT_X+SPH_FLUID_GRID_DIV_SHIFT_Y));
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



    __device__ void updateHash(int i)
    {
        hashes[i] = GridCalculateHash(particles[i].position);
    }

    __device__ void clearGrid(int i)
    {
        grid[i].x = grid[i].y = 0;
    }

    __device__ void updateGrid(int i)
    {
        const uint G_ID = i;
        uint G_ID_PREV = (G_ID == 0)? states[0].fluid_num_particles : G_ID; G_ID_PREV--;
        uint G_ID_NEXT = G_ID + 1; if (G_ID_NEXT == SPH_MAX_FLUID_PARTICLES) { G_ID_NEXT = 0; }
    
        uint cell = hashes[G_ID];
        uint cell_prev = hashes[G_ID_PREV];
        uint cell_next = hashes[G_ID_NEXT];
        if (cell != cell_prev)
        {
            // I'm the start of a cell
            grid[cell].x = G_ID;
        }
        if (cell != cell_next)
        {
            // I'm the end of a cell
            grid[cell].y = G_ID + 1;
        }
    }

    __device__ void computeDensity(int i)
    {
        const uint P_ID = i;
        const float h_sq = d_params.smooth_len * d_params.smooth_len;
        float4 P_position = particles[P_ID].position;

        float density = 0.0f;

        int3 G_XYZ = GridCalculateCell( P_position );
        for(int Z = max(G_XYZ.z - 1, 0) ; Z <= min(G_XYZ.z + 1, SPH_FLUID_GRID_DIV_Z-1) ; Z++)
        {
            for(int Y = max(G_XYZ.y - 1, 0) ; Y <= min(G_XYZ.y + 1, SPH_FLUID_GRID_DIV_Y-1) ; Y++)
            {
                for(int X = max(G_XYZ.x - 1, 0) ; X <= min(G_XYZ.x + 1, SPH_FLUID_GRID_DIV_X-1) ; X++)
                {
                    sphHash G_CELL = GridConstuctKey(make_int3(X, Y, Z));
                    sphGridData G_START_END = grid[G_CELL];
                    for(uint N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
                    {
                        float4 N_position = particles[N_ID].position;
                
                        float4 diff = N_position - P_position;
                        float r_sq = dot(diff, diff);
                        if(r_sq < h_sq)
                        {
                            density += CalculateDensity(r_sq);
                        }
                    }
                }
            }
        }

        forces[P_ID].density = density;
    }

    __device__ void computeForce(int i)
    {
        const uint P_ID = i;
    
        float4 P_position = particles[P_ID].position;
        float4 P_velocity = particles[P_ID].velocity;
        float P_density = forces[P_ID].density;
        float P_pressure = CalculatePressure(P_density);

        const float h_sq = d_params.smooth_len * d_params.smooth_len;

        float4 acceleration = make_float4(0);

        // Calculate the acceleration based on all neighbors
        int3 G_XYZ = GridCalculateCell( P_position );
        for(int Z = max(G_XYZ.z - 1, 0) ; Z <= min(G_XYZ.z + 1, SPH_FLUID_GRID_DIV_Z-1) ; Z++) {
            for(int Y = max(G_XYZ.y - 1, 0) ; Y <= min(G_XYZ.y + 1, SPH_FLUID_GRID_DIV_Y-1) ; Y++) {
                for(int X = max(G_XYZ.x - 1, 0) ; X <= min(G_XYZ.x + 1, SPH_FLUID_GRID_DIV_X-1) ; X++) {
                    sphHash G_CELL = GridConstuctKey(make_int3(X, Y, Z));
                    sphGridData G_START_END = grid[G_CELL];
                    for(uint N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++) {
                        float4 N_position = particles[N_ID].position;
                        float4 diff = N_position - P_position;
                        float r_sq = dot(diff, diff);
                        if(r_sq < h_sq && P_ID != N_ID) {
                            float4 N_velocity = particles[N_ID].velocity;
                            float N_density = forces[N_ID].density;
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

        forces[P_ID].acceleration = acceleration / P_density;
    }

    __device__ void countAlives(int i)
    {
        const uint G_ID = i;
        uint G_ID_NEXT = G_ID + 1; if (G_ID_NEXT == SPH_MAX_FLUID_PARTICLES) { G_ID_NEXT--; }

        if(dead[G_ID] != dead[G_ID_NEXT]) {
            states[0].fluid_num_particles = G_ID + 1;
        }
    }
};

struct FluidDataSet
{
    thrust::device_vector<sphGridParam>     params;
    thrust::device_vector<sphStates>        states;
    thrust::device_vector<sphFluidParticle> particles;
    thrust::device_vector<sphFluidForce>    forces;
    thrust::device_vector<sphHash>          hashes;
    thrust::device_vector<sphDeadFlag>      dead;
    thrust::device_vector<sphFluidMessage>  message;
    thrust::device_vector<sphGridData>      grid;

    FluidDataSet()
    {
        params.resize(1);
        states.resize(1);
        grid.resize(SPH_FLUID_GRID_DIV_3);

        particles.reserve(SPH_MAX_FLUID_PARTICLES);
        forces.reserve(SPH_MAX_FLUID_PARTICLES);
        hashes.reserve(SPH_MAX_FLUID_PARTICLES);
        dead.reserve(SPH_MAX_FLUID_PARTICLES);
        message.reserve(SPH_MAX_FLUID_PARTICLES);
    }
    
    void resizeParticles(size_t n)
    {
        particles.resize(n);
        forces.resize(n);
        hashes.resize(n);
        dead.resize(n);
        message.resize(n);
    }

    DeviceFluidDataSet getDeviceData()
    {
        DeviceFluidDataSet ddata;
        ddata.params    = params.data().get();
        ddata.states    = states.data().get();
        ddata.particles = particles.data().get();
        ddata.forces    = forces.data().get();
        ddata.hashes    = hashes.data().get();
        ddata.dead      = dead.data().get();
        ddata.message   = message.data().get();
        ddata.grid      = grid.data().get();
        return ddata;
    }
};



struct _FluidUpdateHash
{
    DeviceFluidDataSet dfd;
    _FluidUpdateHash(const DeviceFluidDataSet& v) : dfd(v) {}
    __device__ void operator()(int i) { dfd.updateHash(i); }
};

struct _FluidClearGrid
{
    DeviceFluidDataSet dfd;
    _FluidClearGrid(const DeviceFluidDataSet& v) : dfd(v) {}
    __device__ void operator()(int i) { dfd.clearGrid(i); }
};

struct _FluidUpdateGrid
{
    DeviceFluidDataSet dfd;
    _FluidUpdateGrid(const DeviceFluidDataSet& v) : dfd(v) {}
    __device__ void operator()(int i) { dfd.updateGrid(i); }
};

struct _FluidComputeDensity
{
    DeviceFluidDataSet dfd;
    _FluidComputeDensity(const DeviceFluidDataSet& v) : dfd(v) {}
    __device__ void operator()(int i) { dfd.computeDensity(i); }
};

struct _FluidComputeForce
{
    DeviceFluidDataSet dfd;
    _FluidComputeForce(const DeviceFluidDataSet& v) : dfd(v) {}
    __device__ void operator()(int i) { dfd.computeForce(i); }
};

struct _FluidCountAlives
{
    DeviceFluidDataSet dfd;
    _FluidCountAlives(const DeviceFluidDataSet& v) : dfd(v) {}
    __device__ void operator()(int i) { dfd.countAlives(i); }
};


struct _FluidIntegrate
{
    DeviceFluidDataSet dfd;
    DeviceRigidDataSet drd;
    DeviceForceDataSet dgd;
    _FluidIntegrate(const DeviceFluidDataSet &f, const DeviceRigidDataSet &r, const DeviceForceDataSet &g) : dfd(f), drd(r), dgd(g) {}

    __device__ void operator()(int i)
    {
        const uint P_ID = i;

        float4 position = dfd.particles[P_ID].position;
        float4 velocity = dfd.particles[P_ID].velocity;
        float4 acceleration = dfd.forces[P_ID].acceleration;
        int dead = 0;
        position.w = 1.0f;
        dfd.message[P_ID].to = 0;

        for(int i=0; i<drd.num_spheres; ++i) {
            const sphRigidSphere &collision = drd.spheres[i];
            if( position.x >= collision.bb.bl.x && position.x <= collision.bb.ur.x &&
                position.y >= collision.bb.bl.y && position.y <= collision.bb.ur.y &&
                position.z >= collision.bb.bl.z && position.z <= collision.bb.ur.z )
            {
                float r = collision.pos_r.w;
                float r_sq = r*r;
                float4 diff = position - collision.pos_r;
                diff.w = 0.0f;
                float d_sq = dot(diff, diff);
                if(d_sq < r_sq) {
                    float d = sqrt(d_sq);
                    float4 dir = diff / d;
                    acceleration += (d-r) * -d_params.wall_stiffness * dir;
                    dfd.message[P_ID].to = collision.ghandle;
                }
            }
        }
        for(int i=0; i<drd.num_boxes; ++i) {
            const sphRigidBox &collision = drd.boxes[i];
            if( position.x >= collision.bb.bl.x && position.x <= collision.bb.ur.x &&
                position.y >= collision.bb.bl.y && position.y <= collision.bb.ur.y &&
                position.z >= collision.bb.bl.z && position.z <= collision.bb.ur.z )
            {
                int inside = 0;
                int closest_index = 0;
                float closest_dinstance = -9999.0f;
                float4 ppos = position-collision.position;
                for(int p=0; p<6; ++p) {
                    float d = dot(ppos, collision.planes[p]);
                    if(d <= 0.0f) {
                        ++inside;
                        if(d > closest_dinstance) {
                            closest_dinstance = d;
                            closest_index = p;
                        }
                    }
                }
                if(inside==6) {
                    float4 dir = collision.planes[closest_index];
                    dir.w = 0.0f;
                    acceleration += closest_dinstance * -d_params.wall_stiffness * dir;
                    dfd.message[P_ID].to = collision.ghandle;
                }
            }
        }

        dfd.particles[P_ID].energy -= 0.005f;
        if(dfd.message[P_ID].to!=0) {
            float3 vel3 = make_float3(velocity);
            float vl = length(vel3);
            dfd.message[P_ID].velocity3 = vel3;
            dfd.particles[P_ID].energy -= vl;
            if( dfd.particles[P_ID].energy <= 0.0f) {
                dead = 1;
            }
        }

        for(int i=0; i<drd.num_planes; ++i) {
            float4 plane = drd.planes[i].plane;
            float dist = dot(make_float4(position.x, position.y, position.z, 1.0f), plane);
            acceleration += min(dist, 0.0f) * -d_params.wall_stiffness * make_float4(plane.x, plane.y, plane.z, 0.0f);
        }
        // z bound
        acceleration += min(position.z, 0.0f) * -d_params.wall_stiffness * make_float4(0.0f, 0.0f, 1.0f, 0.0f);
        // z gravity
        acceleration += make_float4(0.0f, 0.0f, -8.0f, 0.0f);


        // Apply gravity
        for(int i=0; i<dgd.num_point_gravity; ++i) {
            const float4 center = dgd.point_gravity[i].position;
            const float gravity_strength = dgd.point_gravity[i].strength;

            float4 diff = center-position;
            diff.w = 0.0f;
            float distance = length(diff);
            float4 dir = diff/distance;
            float4 gravity = dir * gravity_strength;

            acceleration += gravity;
        }

        //const float timestep = 1.0f/60.f;
        const float timestep = 0.01f;

        // Integrate
        velocity += timestep * acceleration;
        velocity *= make_float4(0.999);
        {
            float d = dot(velocity, velocity);
            if(d > 1.0f) {
                velocity *= 0.99f;
                if(d > 16.0f) {
                    velocity = velocity / sqrt(d) * 4.0f;
                }
            }
        }
        if(dot(velocity, velocity) > 1.0f) { velocity *= make_float4(0.98); }
        //velocity.z *= 0.0f;
        position += timestep * velocity;
        //position.z *= 0.0f;

        // Update
        dfd.particles[P_ID].density = dfd.forces[P_ID].density;
        dfd.particles[P_ID].position = position;
        dfd.particles[P_ID].velocity = velocity;

        if(dead) {
            dfd.dead[P_ID] = 1;
        }
        else {
            dfd.dead[P_ID] = 0;
            dfd.states[0].fluid_alive_any = 1;
       }
    }
};
