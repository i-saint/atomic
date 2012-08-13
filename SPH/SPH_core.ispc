// コンパイルは↓みたいなカスタムビルドステップで。
// ispc %(FullPath) -o $(TargetDir)%(Filename).obj -h $(TargetDir)%(Filename)_ispc.h --target=sse2,sse4,avx --arch=x86-64 --opt=fast-masked-vload --opt=fast-math
// 
// Intel SPMD はこちら
// http://ispc.github.com/


#include "ispc_vectormath.h"
#include "SPH_const.h"
#include "SPH_collision.h"
typedef unsigned int32 uint32;


#define SPH_PRESSURE_STIFFNESS 200.0f
#define SPH_REST_DENSITY 1000.0f
#define SPH_WALL_STIFFNESS 3000.0f
#define SPH_PARTICLE_MASS 0.002f
#define SPH_PARTICLE_VISCOSITY 0.1f

#define IMP_PARTICLE_STIFFNESS 0.2f
#define IMP_WALL_STIFFNESS 0.5f


static uniform float SPH_DENSITY_COEF;
static uniform float SPH_GRAD_PRESSURE_COEF;
static uniform float SPH_LAP_VISCOSITY_COEF;

export void sphInitializeConstants()
{
    SPH_DENSITY_COEF = SPH_PARTICLE_MASS * 315.0f / (64.0f * PI * pow(SPH_PARTICLE_SIZE, 9));
    SPH_GRAD_PRESSURE_COEF = SPH_PARTICLE_MASS * -45.0f / (PI * pow(SPH_PARTICLE_SIZE, 6));
    SPH_LAP_VISCOSITY_COEF = SPH_PARTICLE_MASS * SPH_PARTICLE_VISCOSITY * 45.0f / (PI * pow(SPH_PARTICLE_SIZE, 6));
}

struct Vec3 { float x,y,z; };
static soa<8> Vec3 g_accel[SPH_MAX_PARTICLE_NUM];



static uniform Sphere *g_spheres;
static uniform int32 g_num_spheres;

static uniform Plane *g_planes;
static uniform int32 g_num_planes;

static uniform Box *g_boxes;
static uniform int32 g_num_boxes;

export void sphSetCollisionShapes(
    uniform Sphere spheres[], uniform int32 num_spheres,
    uniform Plane planes[], uniform int32 num_planes,
    uniform Box boxes[], uniform int32 num_boxes )
{
    g_spheres=spheres; g_num_spheres=num_spheres;
    g_planes=planes; g_num_planes=num_planes;
    g_boxes=boxes; g_num_boxes=num_boxes;
}





static inline float sphComputeDensity(
    vec3 pos1,
    vec3 pos2 )
{
    static uniform const float h_sq = SPH_PARTICLE_SIZE * SPH_PARTICLE_SIZE;
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
    uniform const GridData &gd = grid[yi*SPH_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];

    uniform const int32 nx_beg = max(xi-1, 0);
    uniform const int32 nx_end = min(xi+1, SPH_GRID_DIV-1);
    uniform const int32 ny_beg = max(yi-1, 0);
    uniform const int32 ny_end = min(yi+1, SPH_GRID_DIV-1);

    for(uniform int32 i=0; i<particle_num; ++i) {
        uniform vec3 pos1 = {particles[i].x, particles[i].y, particles[i].z};
        float density = 0.0f;
        for(uniform int32 nyi=ny_beg; nyi<=ny_end; ++nyi) {
            for(uniform int32 nxi=nx_beg; nxi<=nx_end; ++nxi) {
                uniform const GridData &ngd = grid[nyi*SPH_GRID_DIV + nxi];
                uniform const int32 neighbor_num = ngd.end - ngd.begin;
                soa<8> Particle * uniform neighbors = &all_particles[ngd.soai*8];
                foreach(t=0 ... neighbor_num) {
                    vec3 pos2 = {neighbors[t].x, neighbors[t].y, neighbors[t].z};
                    density += sphComputeDensity(pos1, pos2);
                }
            }
        }

        particles[i].density = reduce_add(density);
    }
}


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
    const float h = SPH_PARTICLE_SIZE;
    float avg_pressure = 0.5f * (N_pressure + P_pressure);
    // Implements this equation:
    // W_spkiey(r, h) = 15 / (pi * h^6) * (h - r)^3
    // GRAD( W_spikey(r, h) ) = -45 / (pi * h^6) * (h - r)^2
    // g_fGradPressureCoef = fParticleMass * -45.0f / (PI * fSmoothlen^6)
    return (SPH_GRAD_PRESSURE_COEF * avg_pressure / N_density * (h - r) * (h - r) / r) * diff;
}

static inline vec3 sphCalculateLapVelocity(float r, vec3 P_velocity, vec3 N_velocity, float N_density)
{
    const float h = SPH_PARTICLE_SIZE;
    vec3 vel_diff = N_velocity - P_velocity;
    // Implements this equation:
    // W_viscosity(r, h) = 15 / (2 * pi * h^3) * (-r^3 / (2 * h^3) + r^2 / h^2 + h / (2 * r) - 1)
    // LAPLACIAN( W_viscosity(r, h) ) = 45 / (pi * h^6) * (h - r)
    // g_fLapViscosityCoef = fParticleMass * fViscosity * 45.0f / (PI * fSmoothlen^6)
    return  (SPH_LAP_VISCOSITY_COEF / N_density * (h - r)) * vel_diff;
}

static inline float sphCalculateDensity(float r_sq)
{
    const float h_sq = SPH_PARTICLE_SIZE * SPH_PARTICLE_SIZE;
    // Implements this equation:
    // W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
    // g_fDensityCoef = fParticleMass * 315.0f / (64.0f * PI * fSmoothlen^9)
    return SPH_DENSITY_COEF * (h_sq - r_sq) * (h_sq - r_sq) * (h_sq - r_sq);
}

static inline vec3 sphComputeAccel(
    vec3 pos1,
    vec3 pos2,
    vec3 vel1,
    vec3 vel2,
    float pressure1,
    float density2 )
{
    static uniform const float h_sq = SPH_PARTICLE_SIZE * SPH_PARTICLE_SIZE;
    vec3 accel = {0.0f, 0.0f, 0.0f};
    vec3 diff = pos2 - pos1;
    float r_sq = dot3(diff, diff);
    if(r_sq < h_sq && r_sq > 0.0f) {
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
    uniform const GridData &gd = grid[yi*SPH_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Vec3 * uniform accels = &g_accel[gd.soai*8];

    uniform const int32 nx_beg = max(xi-1, 0);
    uniform const int32 nx_end = min(xi+1, SPH_GRID_DIV-1);
    uniform const int32 ny_beg = max(yi-1, 0);
    uniform const int32 ny_end = min(yi+1, SPH_GRID_DIV-1);

    for(uniform int32 i=0; i<particle_num; ++i) {
        uniform vec3 pos1 = {particles[i].x, particles[i].y, particles[i].z};
        uniform vec3 vel1 = {particles[i].vx, particles[i].vy, particles[i].vz};
        uniform float density1 = particles[i].density;
        uniform float pressure1 = sphCalculatePressure(density1);

        vec3 accel = {0.0f, 0.0f, 0.0f};
        for(uniform int32 nyi=ny_beg; nyi<=ny_end; ++nyi) {
            for(uniform int32 nxi=nx_beg; nxi<=nx_end; ++nxi) {
                uniform const GridData &ngd = grid[nyi*SPH_GRID_DIV + nxi];
                uniform const int32 neighbor_num = ngd.end - ngd.begin;
                soa<8> Particle * uniform neighbors = &all_particles[ngd.soai*8];
                foreach(t=0 ... neighbor_num) {
                    vec3 pos2 = {neighbors[t].x, neighbors[t].y, neighbors[t].z};
                    vec3 vel2 = {neighbors[t].vx, neighbors[t].vy, neighbors[t].vz};
                    float density2 = neighbors[t].density;
                    accel += sphComputeAccel(pos1, pos2, vel1, vel2, pressure1, density2);
                }
            }
        }

        accels[i].x = reduce_add(accel.x);
        accels[i].y = reduce_add(accel.y);
        accels[i].z = reduce_add(accel.z);
    }
}


static inline void sphComputeGravity(
    vec3 pos1,
    vec3 &vel1 )
{
    const uniform float CENTER_SPHERE_RADIUS = 1.75f;
    const uniform float GRAVITY = 5.0f;

    {
        // 中心の球との衝突判定
        vec3 diff = -pos1;
        float d = length3(diff);
        vec3 dir = diff / d;
        vel1 += dir * (min(0.0f, d-CENTER_SPHERE_RADIUS) * SPH_WALL_STIFFNESS);

        // 重力加速
        vel1 += dir * GRAVITY;
    }

    //uniform const float<4> planes[6] = {
    //    { 0.0f, 1.0f, 0.0f,-0},
    //    { 0.0f,-1.0f, 0.0f, SPH_GRID_SIZE},
    //    { 1.0f, 0.0f, 0.0f,-SPH_GRID_SIZE},
    //    {-1.0f, 0.0f, 0.0f, SPH_GRID_SIZE},
    //    { 0.0f, 0.0f, 1.0f,-SPH_GRID_SIZE},
    //    { 0.0f, 0.0f,-1.0f, SPH_GRID_SIZE},
    //};
    //for(int i=0; i<1; ++i) {
    //    vec3 n = planes[i].xyz;
    //    float d = dot3(n, pos1) + planes[i].w;
    //    vel1 += n * (-min(0.0f, d) * SPH_WALL_STIFFNESS);
    //}
}



export void sphIntegrate(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform const GridData &gd = grid[yi*SPH_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];
    soa<8> Vec3 * uniform accels = &g_accel[gd.soai*8];

    uniform const float timestep = 0.01f;

    foreach(i=0 ... particle_num) {
        vec3 vel = {particles[i].vx, particles[i].vy, particles[i].vz};
        vec3 accel = {accels[i].x, accels[i].y, accels[i].z};
        //vec3 accel = {0.0f, 0.0f, 0.0f};
        vec3 pos = {particles[i].x, particles[i].y, particles[i].z};

        const uniform float DECELERATE = 0.999;
        sphComputeGravity(pos, accel);
        vel += accel * timestep;
        // 自然減速
        vel *= DECELERATE;

        pos += vel * timestep;

        particles[i].vx = vel.x;
        particles[i].vy = vel.y;
        particles[i].vz = vel.z;

        particles[i].x = pos.x;
        particles[i].y = pos.y;
        particles[i].z = pos.z;
    }
}




static inline void impComputeParticleInteraction(
    vec3 pos1,
    vec3 pos2,
    vec3 &accel )
{
    const uniform float SPH_RCP_PARTICLE_SIZE2 = 1.0f/(SPH_PARTICLE_SIZE*2.0f);
    vec3 diff = pos2 - pos1;
    float d = length3(diff); // float d = length3est(diff); // 何故か est の方が遅くなる
    if(d > 0.0f) { // d==0: 衝突元＆衝突相手が同一
        vec3 dir = diff * SPH_RCP_PARTICLE_SIZE2; // vec3 dir = diff / d;
        accel += dir * (min(0.0f, d-(SPH_PARTICLE_SIZE*2.0f)) * IMP_PARTICLE_STIFFNESS);
    }
}

static inline void impComputeGravity(
    vec3 pos1,
    vec3 &vel1 )
{
    // x,y,z=0,0,0 で半径 CENTER_SPHERE_RADIUS の球があるもんとして重力で引っ張る＆衝突判定

    const uniform float CENTER_SPHERE_RADIUS = 3.0f;
    const uniform float GRAVITY = 0.002f;

    {
        // 中心の球との衝突判定
        vec3 diff = -pos1;
        float d = length3(diff);
        vec3 dir = diff / d;
        vel1 += dir * (min(0.0f, d-CENTER_SPHERE_RADIUS) * IMP_WALL_STIFFNESS);

        // 重力加速
        vel1 += dir * GRAVITY;
    }

    //const uniform float<4> planes[6] = {
    //    { 0.0f, 0.0f, 1.0f,-SPH_GRID_SIZE},
    //    { 0.0f, 0.0f,-1.0f, SPH_GRID_SIZE},
    //    { 0.0f, 1.0f, 0.0f,-SPH_GRID_SIZE},
    //    { 0.0f,-1.0f, 0.0f, SPH_GRID_SIZE},
    //    { 1.0f, 0.0f, 0.0f,-SPH_GRID_SIZE},
    //    {-1.0f, 0.0f, 0.0f, SPH_GRID_SIZE},
    //};
    //for(int i=0; i<0; ++i) {
    //    vec3 n = planes[i].xyz;
    //    float d = dot3(n, pos1) + planes[i].w;
    //    vel1 += n * (min(0.0f, d) * IMP_WALL_STIFFNESS);
    //}
}

export void impUpdateVelocity(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform const GridData &gd = grid[yi*SPH_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];

    uniform const int32 nx_beg = max(xi-1, 0);
    uniform const int32 nx_end = min(xi+1, SPH_GRID_DIV-1);
    uniform const int32 ny_beg = max(yi-1, 0);
    uniform const int32 ny_end = min(yi+1, SPH_GRID_DIV-1);

    for(uniform int32 i=0; i<particle_num; ++i) {
        uniform vec3 pos1 = {particles[i].x, particles[i].y, particles[i].z};
        uniform vec3 vel1 = {particles[i].vx, particles[i].vy, particles[i].vz};
        vec3 accel = {0.0f, 0.0f, 0.0f};
        for(uniform int32 nyi=ny_beg; nyi<=ny_end; ++nyi) {
            for(uniform int32 nxi=nx_beg; nxi<=nx_end; ++nxi) {
                uniform const GridData &ngd = grid[nyi*SPH_GRID_DIV + nxi];
                soa<8> Particle * uniform neighbors = &all_particles[ngd.soai*8];
                uniform const int32 neighbor_num = ngd.end - ngd.begin;
                foreach(t=0 ... neighbor_num) {
                    vec3 pos2 = {neighbors[t].x, neighbors[t].y, neighbors[t].z};
                    impComputeParticleInteraction(pos1, pos2, accel);
                }
            }
        }

        vel1.x  += reduce_add(accel.x);
        vel1.y  += reduce_add(accel.y);
        vel1.z  += reduce_add(accel.z);
        particles[i].vx = vel1.x;
        particles[i].vy = vel1.y;
        particles[i].vz = vel1.z;
    }
    foreach(i=0 ... particle_num) {
        vec3 pos1 = {particles[i].x, particles[i].y, particles[i].z};
        vec3 vel1 = {particles[i].vx, particles[i].vy, particles[i].vz};

        impComputeGravity(pos1, vel1);

        // 自然減速
        const uniform float DECELERATE = 0.995;
        vel1 *= DECELERATE;

        particles[i].vx = vel1.x;
        particles[i].vy = vel1.y;
        particles[i].vz = vel1.z;
    }
}

export void impIntegrate(
    soa<8> Particle all_particles[],
    GridData uniform grid[],
    uniform int32 xi, uniform int32 yi )
{
    uniform const GridData &gd = grid[yi*SPH_GRID_DIV + xi];
    uniform const int32 particle_num = gd.end - gd.begin;
    soa<8> Particle * uniform particles = &all_particles[gd.soai*8];

    foreach(i=0 ... particle_num) {
        vec3 pos = {particles[i].x, particles[i].y, particles[i].z};
        vec3 vel = {particles[i].vx, particles[i].vy, particles[i].vz};
        pos += vel;
        particles[i].x = pos.x;
        particles[i].y = pos.y;
        particles[i].z = pos.z;
    }
}
