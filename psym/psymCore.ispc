// コンパイルは↓みたいなカスタムビルドステップで。
// ispc %(FullPath) -o $(TargetDir)%(Filename).obj -h $(TargetDir)%(Filename)_ispc.h --target=sse2,sse4,avx --arch=x86-64 --opt=fast-masked-vload --opt=fast-math
// 
// Intel SPMD はこちら
// http://ispc.github.com/


#include "ispc_vectormath.h"
#include "ispc_collision.h"
#include "psymConst.h"


#define SPH_SMOOTH_LEN          0.02f
#define SPH_PRESSURE_STIFFNESS  50.0f
#define SPH_REST_DENSITY        500.0f
#define SPH_PARTICLE_MASS       0.001f
#define SPH_PARTICLE_VISCOSITY  0.2f
#define SPH_WALL_STIFFNESS      3000.0f

#define IMP_PRESSURE_STIFFNESS 50.0f

#define SPH_TIMESTEP 0.01f
#define SPH_DECELERATE 0.996f


static uniform float SPH_DENSITY_COEF;
static uniform float SPH_GRAD_PRESSURE_COEF;
static uniform float SPH_LAP_VISCOSITY_COEF;

export void sphInitializeConstants()
{
    SPH_DENSITY_COEF = SPH_PARTICLE_MASS * 315.0f / (64.0f * PI * pow(SPH_SMOOTH_LEN, 9));
    SPH_GRAD_PRESSURE_COEF = SPH_PARTICLE_MASS * -45.0f / (PI * pow(SPH_SMOOTH_LEN, 6));
    SPH_LAP_VISCOSITY_COEF = SPH_PARTICLE_MASS * SPH_PARTICLE_VISCOSITY * 45.0f / (PI * pow(SPH_SMOOTH_LEN, 6));
}

static soa<8> Force g_forces[PSYM_MAX_PARTICLE_NUM];



#define repulse(p, f, n, d, shape)\
    {\
        p.hit_to = shape.id;\
        vec3 accel = get_accel(f);\
        accel += n * (-d * SPH_WALL_STIFFNESS);\
        set_accel(f, accel);\
    }

export void Dummy(uniform Plane planes[]) {}

export void sphProcessCollision(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi,
    uniform RigidSphere spheres[], uniform int32 num_spheres,
    uniform RigidPlane planes[], uniform int32 num_planes,
    uniform RigidBox boxes[], uniform int32 num_boxes )
{
    uniform const GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    if(particle_num==0) { return; }
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Force * uniform forces = &g_forces[gd.soai*8];

    uniform float particle_radius = SPH_SMOOTH_LEN;
    uniform float cell_size = PSYM_GRID_SIZE / PSYM_GRID_DIV;
    uniform vec2 grid_bl = {PSYM_GRID_POS + cell_size*xi, PSYM_GRID_POS + cell_size*yi };
    uniform vec2 grid_ur = {PSYM_GRID_POS + cell_size*(xi+1), PSYM_GRID_POS + cell_size*(yi+1) };

    foreach(i=0 ... particle_num) {
        particles[i].hit_to = 0;
    }

    // 適切に関数に分けたかったが、struct の参照渡しができないとか謎のコンパイラクラッシュとかで断念。
    // 全部 inline で書きます。

    // RigidPlane
    for(uniform int32 s=0; s<num_planes; ++s) {
        uniform const RigidPlane &shape = planes[s];
        uniform const vec3 plane_normal = get_normal(shape);
        uniform const float plane_distance = shape.distance;
        foreach(i=0 ... particle_num) {
            vec3 particles_pos = get_pos(particles[i]);
            float distance = dot3(particles_pos, plane_normal) + plane_distance;
            if(distance < 0.0f) {
                repulse(particles[i], forces[i], plane_normal, distance, shape);
            }
        }
    }

    // RigidSphere
    for(uniform int32 s=0; s<num_spheres; ++s) {
        uniform const RigidSphere &shape = spheres[s];
        uniform vec3 bb_bl = get_bl(shape.bb);
        uniform vec3 bb_ur = get_ur(shape.bb);
        if(grid_ur.x < bb_bl.x || grid_ur.y < bb_bl.y ||
           grid_bl.x > bb_ur.x || grid_bl.y > bb_ur.y )
        {
            continue;
        }

        uniform const vec3 shape_pos = get_pos(shape);
        uniform const float r = shape.radius;
        uniform const float r_sq = r*r;
        foreach(i=0 ... particle_num) {
            vec3 particles_pos = get_pos(particles[i]);
            vec3 diff = shape_pos - particles_pos;
            float d_sq = length_sq3(diff);
            if(d_sq < r_sq) {
                float d = sqrt(d_sq);
                vec3 dir = diff / d;
                float pierce = d - r;
                repulse(particles[i], forces[i], dir, -pierce, shape);
            }
        }
    }

    // RigidBox
    for(uniform int32 s=0; s<num_boxes; ++s) {
        uniform const RigidBox &shape = boxes[s];
        uniform vec3 bb_bl = get_bl(shape.bb);
        uniform vec3 bb_ur = get_ur(shape.bb);
        if(grid_ur.x < bb_bl.x || grid_ur.y < bb_bl.y ||
           grid_bl.x > bb_ur.x || grid_bl.y > bb_ur.y )
        {
            continue;
        }

        uniform vec3 box_pos = get_pos(shape);
        foreach(i=0 ... particle_num) {
            vec3 particles_pos = get_pos(particles[i]);

            int32 inside = 0;
            float closest_distance = -9999.0f;
            vec3 closest_normal;
            vec3 ppos = particles_pos - box_pos;
            for(uniform int32 p=0; p<6; ++p) {
                uniform const vec3 plane_normal = get_normal(shape.planes[p]);
                uniform const float plane_distance = shape.planes[p].distance;
                float distance = dot3(ppos, plane_normal) + plane_distance;
                if(distance < 0.0f) {
                    inside++;
                    if(distance > closest_distance) {
                        closest_distance = distance;
                        closest_normal = plane_normal;
                    }
                }
            }
            if(inside==6) {
                repulse(particles[i], forces[i], closest_normal, closest_distance, shape);
            }
        }
    }
}
#undef repulse

export void sphProcessExternalForce(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi,
    uniform PointForce pforce[], uniform int32 num_pforce,
    uniform DirectionalForce dforce[], uniform int32 num_dforce,
    uniform BoxForce bforce[], uniform int32 num_bforce )
{
    uniform const GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    if(particle_num==0) { return; }
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Force * uniform forces = &g_forces[gd.soai*8];

    uniform float particle_radius = SPH_SMOOTH_LEN;

    // Point
    for(uniform int32 s=0; s<num_pforce; ++s) {
        uniform const PointForce &f = pforce[s];
        uniform const vec3 force_pos = get_pos(f);
        uniform const float strength = f.strength;
        foreach(i=0 ... particle_num) {
            vec3 particles_pos = get_pos(particles[i]);
            vec3 diff = particles_pos - force_pos;
            vec3 dir = normalize3(diff);
            vec3 accel = get_accel(forces[i]);
            accel -= dir * strength;
            set_accel(forces[i], accel);
        }
    }

    // Directional
    for(uniform int32 s=0; s<num_dforce; ++s) {
        uniform const DirectionalForce &f = dforce[s];
        uniform const vec3 dir = get_normal(f);
        uniform const float strength = f.strength;
        foreach(i=0 ... particle_num) {
            vec3 accel = get_accel(forces[i]);
            accel += dir * strength;
            set_accel(forces[i], accel);
        }
    }

    // todo:
    //// RigidBox
    //for(uniform int32 s=0; s<num_bforce; ++s) {
    //}
}





static inline float sphComputeDensity(
    vec3 pos1,
    vec3 pos2 )
{
    static uniform const float h_sq = SPH_SMOOTH_LEN * SPH_SMOOTH_LEN;
    vec3 diff = pos2 - pos1;
    float r_sq = dot3(diff, diff);
    if(r_sq < h_sq) {
        // Implements this equation:
        // W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
        // g_fDensityCoef = fParticleMass * 315.0f / (64.0f * PI * fSmoothlen^9)
        return SPH_DENSITY_COEF * (h_sq - r_sq) * (h_sq - r_sq) * (h_sq - r_sq);
    }
    return 0.0f;
}

export void sphUpdateDensity(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];

#ifdef psym_enable_neighbor_density_estimation
    for(uniform int32 i=0; i<particle_num; ++i) {
        uniform vec3 pos1 = get_pos(particles[i]);
        float density = 0.0f;

        uniform const GridData &ngd = gd;
        uniform const int32 neighbor_num = ngd.end - ngd.begin;
        soa<8> Particle * uniform neighbors = &all_particles[ngd.soai*8];
        foreach(t=0 ... neighbor_num) {
            vec3 pos2 = get_pos(neighbors[t]);
            density += sphComputeDensity(pos1, pos2);
        }
        particles[i].density = reduce_add(density);
    }
    {
        gd.density = 0.0f;
        float total_density = 0.0f;
        foreach(i=0 ... particle_num) {
            total_density += particles[i].density;
        }
        gd.density = reduce_add(total_density) / particle_num;
    }
#else
    uniform const int32 nx_beg = max(xi-1, 0);
    uniform const int32 nx_end = min(xi+1, PSYM_GRID_DIV-1);
    uniform const int32 ny_beg = max(yi-1, 0);
    uniform const int32 ny_end = min(yi+1, PSYM_GRID_DIV-1);

    for(uniform int32 i=0; i<particle_num; ++i) {
        uniform vec3 pos1 = get_pos(particles[i]);
        float density = 0.0f;
        for(uniform int32 nyi=ny_beg; nyi<=ny_end; ++nyi) {
            for(uniform int32 nxi=nx_beg; nxi<=nx_end; ++nxi) {
                uniform const GridData &ngd = grid[nyi*PSYM_GRID_DIV + nxi];
                uniform const int32 neighbor_num = ngd.end - ngd.begin;
                soa<8> Particle * uniform neighbors = &all_particles[ngd.soai*8];
                foreach(t=0 ... neighbor_num) {
                    vec3 pos2 = get_pos(neighbors[t]);
                    density += sphComputeDensity(pos1, pos2);
                }
            }
        }
        particles[i].density = reduce_add(density);
    }
#endif // psym_enable_neighbor_density_estimation
}

#ifdef psym_enable_neighbor_density_estimation
export void sphUpdateDensity2(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];

    uniform const int32 nx_beg = max(xi-1, 0);
    uniform const int32 nx_end = min(xi+1, PSYM_GRID_DIV-1);
    uniform const int32 ny_beg = max(yi-1, 0);
    uniform const int32 ny_end = min(yi+1, PSYM_GRID_DIV-1);

    for(uniform int32 nyi=ny_beg; nyi<=ny_end; ++nyi) {
        for(uniform int32 nxi=nx_beg; nxi<=nx_end; ++nxi) {
            uniform const GridData &ngd = grid[nyi*PSYM_GRID_DIV + nxi];
            foreach(i=0 ... particle_num) {
                particles[i].density += ngd.density*0.05f;
            }
        }
    }
}
#endif // psym_enable_neighbor_density_estimation

static inline float sphCalculatePressure(float density)
{
    // Implements this equation:
    // Pressure = B * ((rho / rho_0)^y  - 1)
    return SPH_PRESSURE_STIFFNESS * max(pow(density / SPH_REST_DENSITY, 3) - 1.0f, 0.0f);
}
static inline uniform float sphCalculatePressure(uniform float density)
{
    return SPH_PRESSURE_STIFFNESS * max(pow(density / SPH_REST_DENSITY, 3) - 1.0f, 0.0f);
}


static inline vec3 sphCalculateGradPressure(float r, float P_pressure, float N_pressure, float N_density, vec3 diff)
{
    const float h = SPH_SMOOTH_LEN;
    float avg_pressure = 0.5f * (N_pressure + P_pressure);
    // Implements this equation:
    // W_spkiey(r, h) = 15 / (pi * h^6) * (h - r)^3
    // GRAD( W_spikey(r, h) ) = -45 / (pi * h^6) * (h - r)^2
    // g_fGradPressureCoef = fParticleMass * -45.0f / (PI * fSmoothlen^6)
    return (SPH_GRAD_PRESSURE_COEF * avg_pressure / N_density * (h - r) * (h - r) / r) * diff;
}

static inline vec3 sphCalculateLapVelocity(float r, vec3 P_velocity, vec3 N_velocity, float N_density)
{
    const float h = SPH_SMOOTH_LEN;
    vec3 vel_diff = N_velocity - P_velocity;
    // Implements this equation:
    // W_viscosity(r, h) = 15 / (2 * pi * h^3) * (-r^3 / (2 * h^3) + r^2 / h^2 + h / (2 * r) - 1)
    // LAPLACIAN( W_viscosity(r, h) ) = 45 / (pi * h^6) * (h - r)
    // g_fLapViscosityCoef = fParticleMass * fViscosity * 45.0f / (PI * fSmoothlen^6)
    return  (SPH_LAP_VISCOSITY_COEF / N_density * (h - r)) * vel_diff;
}

static inline vec3 sphComputeAccel(
    vec3 pos1,
    vec3 pos2,
    vec3 vel1,
    vec3 vel2,
    float pressure1,
    float density2)
{
    static uniform const float h_sq = SPH_SMOOTH_LEN * SPH_SMOOTH_LEN;
    vec3 accel = {0.0f, 0.0f, 0.0f};
    vec3 diff = pos2 - pos1;
    float r_sq = dot3(diff, diff);
    if(r_sq < h_sq && r_sq>0.0f) {
        float pressure2 = sphCalculatePressure(density2);
        float r = sqrt(r_sq);

        // Pressure Term
        accel += sphCalculateGradPressure(r, pressure1, pressure2, density2, diff);

        // Viscosity Term
        accel += sphCalculateLapVelocity(r, vel1, vel2, density2);
    }
    return accel;
}

export void sphUpdateForce(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform const GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Force * uniform forces = &g_forces[gd.soai*8];

    uniform const int32 nx_beg = max(xi-1, 0);
    uniform const int32 nx_end = min(xi+1, PSYM_GRID_DIV-1);
    uniform const int32 ny_beg = max(yi-1, 0);
    uniform const int32 ny_end = min(yi+1, PSYM_GRID_DIV-1);

    for(uniform int32 i=0; i<particle_num; ++i) {
        uniform vec3 pos1 = get_pos(particles[i]);
        uniform vec3 vel1 = get_vel(particles[i]);
        uniform float density1 = particles[i].density;
        uniform float pressure1 = sphCalculatePressure(density1);

        vec3 accel = {0.0f, 0.0f, 0.0f};
        for(uniform int32 nyi=ny_beg; nyi<=ny_end; ++nyi) {
            for(uniform int32 nxi=nx_beg; nxi<=nx_end; ++nxi) {
                uniform const GridData &ngd = grid[nyi*PSYM_GRID_DIV + nxi];
                uniform const int32 neighbor_num = ngd.end - ngd.begin;
                soa<8> Particle * uniform neighbors = &all_particles[ngd.soai*8];
                foreach(t=0 ... neighbor_num) {
                    vec3 pos2 = get_pos(neighbors[t]);
                    vec3 vel2 = get_vel(neighbors[t]);
                    float density2 = neighbors[t].density;
                    accel += sphComputeAccel(pos1, pos2, vel1, vel2, pressure1, density2);
                }
            }
        }

        uniform float rcp_density1 = 1.0f / density1;
        forces[i].ax = reduce_add(accel.x) * rcp_density1;
        forces[i].ay = reduce_add(accel.y) * rcp_density1;
        forces[i].az = reduce_add(accel.z) * rcp_density1;
    }
}


export void sphIntegrate(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform const GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Force * uniform forces = &g_forces[gd.soai*8];

    uniform const float timestep = SPH_TIMESTEP;
    foreach(i=0 ... particle_num) {
        vec3 vel = get_vel(particles[i]);
        vec3 accel = get_accel(forces[i]);
        vec3 pos = get_pos(particles[i]);

        vel += accel * timestep;
        // 自然減速
        {
            float d = length3(vel) * 0.075;
            float c = max(1.0f - d*d, 0.01);
            vel *= c;
        }

        pos += vel * timestep;

        set_vel(particles[i], vel);
        set_pos(particles[i], pos);
    }
}




static inline void impComputeParticleInteraction(
    vec3 pos1,
    vec3 pos2,
    vec3 &accel )
{
    const uniform float SPH_RCP_PARTICLE_SIZE2 = 1.0f/(SPH_SMOOTH_LEN*2.0f);
    vec3 diff = pos2 - pos1;    
    float d = length3(diff); // float d = length3est(diff); // 何故か est の方が遅くなる
    if(d > 0.0f) { // d==0: 衝突元＆衝突相手が同一
        vec3 dir = diff * SPH_RCP_PARTICLE_SIZE2; // vec3 dir = diff / d;
        accel += dir * (min(0.0f, d-(SPH_SMOOTH_LEN*2.0f)) * IMP_PRESSURE_STIFFNESS);
    }
}

export void impUpdateVelocity(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform const GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Force * uniform forces = &g_forces[gd.soai*8];

    uniform const int32 nx_beg = max(xi-1, 0);
    uniform const int32 nx_end = min(xi+1, PSYM_GRID_DIV-1);
    uniform const int32 ny_beg = max(yi-1, 0);
    uniform const int32 ny_end = min(yi+1, PSYM_GRID_DIV-1);

    for(uniform int32 i=0; i<particle_num; ++i) {
        uniform vec3 pos1 = get_pos(particles[i]);
        vec3 accel = {0.0f, 0.0f, 0.0f};
        for(uniform int32 nyi=ny_beg; nyi<=ny_end; ++nyi) {
            for(uniform int32 nxi=nx_beg; nxi<=nx_end; ++nxi) {
                uniform const GridData &ngd = grid[nyi*PSYM_GRID_DIV + nxi];
                soa<8> Particle * uniform neighbors = &all_particles[ngd.soai*8];
                uniform const int32 neighbor_num = ngd.end - ngd.begin;
                foreach(t=0 ... neighbor_num) {
                    vec3 pos2 = get_pos(neighbors[t]);
                    impComputeParticleInteraction(pos1, pos2, accel);
                }
            }
        }

        forces[i].ax = reduce_add(accel.x);
        forces[i].ay = reduce_add(accel.y);
        forces[i].az = reduce_add(accel.z);
    }
}

export void impIntegrate(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform const GridData &gd = grid[yi*PSYM_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Force * uniform forces = &g_forces[gd.soai*8];

    uniform const float timestep = SPH_TIMESTEP;

    foreach(i=0 ... particle_num) {
        vec3 pos = get_pos(particles[i]);
        vec3 accel = get_accel(forces[i]);
        vec3 vel = get_vel(particles[i]);

        vel += accel * timestep;
        // 自然減速
        vel *= SPH_DECELERATE;

        pos += vel * timestep;
        set_pos(particles[i], pos);
        set_vel(particles[i], vel);
    }
}
