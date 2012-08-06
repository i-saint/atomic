#ifndef __SPH_collision_h__
#define __SPH_collision_h__

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
    float   posx, posy, posz;
    float   velx, vely, velz;
    float   density;
};




inline void TestCollide(
    soa<8> Particle particles[],
    uniform int32 particle_num,
    uniform Sphere shape )
{
    foreach(i=0 ... particle_num) {
    }
}


inline void TestCollide(
    soa<8> Particle particles[],
    uniform int32 particle_num,
    uniform Plane shape )
{
    foreach(i=0 ... particle_num) {
    }
}


inline void TestCollide(
    soa<8> Particle particles[],
    uniform int32 particle_num,
    uniform Box shape )
{
    foreach(i=0 ... particle_num) {
    }
}

#endif // __SPH_collision_h__
