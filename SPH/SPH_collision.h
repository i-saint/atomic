#ifndef __SPH_collision_h__
#define __SPH_collision_h__

#include "ispc_vectormath.h"
#include "SPH_const.h"


struct Sphere
{
    float x, y, z;
    float radius;
};

struct Plane
{
    float nx, ny, nz;
    float distance;
};

struct Box
{
    Plane planes[6];
};


struct Particle
{
    float   x, y, z;
    float   vx, vy, vz;
    float   density;
    int16   hit;
    int16   flag_live;
};

struct GridData
{
    int32 begin, end;
    int32 soai;
    float density;
};




inline void TestCollide(
    soa<8> Particle particles[],
    uniform int32 particle_num,
    uniform Sphere shape )
{
    uniform vec3 sphere_pos = {shape.x, shape.y, shape.z};
    uniform float sphere_radius_sq = shape.radius * shape.radius;
    uniform float particle_radius_sp = SPH_PARTICLE_SIZE*SPH_PARTICLE_SIZE;
    foreach(i=0 ... particle_num) {
        vec3 particles_pos = {particles[i].x, particles[i].y, particles[i].z};
        vec3 diff = sphere_pos - particles_pos;
        float distance_sq = dot3(diff, diff);
        if(distance_sq <= particle_radius_sp) {
            particles[i].hit++;
        }
    }
}


inline void TestCollide(
    soa<8> Particle particles[],
    uniform int32 particle_num,
    uniform Plane shape )
{
    uniform vec3 plane_normal = {shape.nx, shape.ny, shape.nz};
    uniform float plane_distance = shape.distance;
    uniform float particle_radius = SPH_PARTICLE_SIZE;
    foreach(i=0 ... particle_num) {
        vec3 particles_pos = {particles[i].x, particles[i].y, particles[i].z};
        float d = dot3(particles_pos, (varying vec3)plane_normal);
        if(d <= particle_radius) {
            particles[i].hit++;
        }
    }
}


inline void TestCollide(
    soa<8> Particle particles[],
    uniform int32 particle_num,
    uniform Box shape )
{
    uniform float particle_radius = SPH_PARTICLE_SIZE;
    foreach(i=0 ... particle_num) {
        vec3 particles_pos = {particles[i].x, particles[i].y, particles[i].z};
        int32 inside = 0;
        for(uniform int32 p=0; p<6; ++p) {
            uniform vec3 plane_normal = {shape.planes[p].nx, shape.planes[p].ny, shape.planes[p].nz};
            uniform float plane_distance = shape.planes[p].distance;
            float d = dot3(particles_pos, (varying vec3)plane_normal);
            if(d <= particle_radius) {
                inside++;
            }
        }
        if(inside==6) {
            particles[i].hit++;
        }
    }
}

#endif // __SPH_collision_h__
