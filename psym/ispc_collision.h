#ifndef __SPH_collision_h__
#define __SPH_collision_h__

#include "ispc_vectormath.h"
#include "psymConst.h"
typedef unsigned int32 uint32;


struct BoundingBox
{
    float bl_x, bl_y, bl_z;
    float ur_x, ur_y, ur_z;
};
struct Plane {
    float nx, ny, nz;
    float distance;
};


struct RigidSphere
{
    uint32 id;
    BoundingBox bb;
    float x, y, z;
    float radius;
};

struct RigidBox
{
    uint32 id;
    BoundingBox bb;
    float x, y, z;
    Plane planes[6];
};

struct RigidPlane
{
    uint32 id;
    BoundingBox bb;
    float nx, ny, nz;
    float distance;
};


struct PointForce
{
    float x, y, z;
    float strength;
};

struct DirectionalForce
{
    float nx, ny, nz;
    float strength;
};

struct BoxForce
{
    float nx, ny, nz;
    float strength;
    Plane planes[6];
};


struct Particle
{
    float   x, y, z;
    float   vx, vy, vz;
    float   density;
    uint32  hit_to;
};

struct Force
{
    float ax, ay, az;
};

struct GridData
{
    int32 begin, end;
    int32 soai;
    float density;
};


// struct ‚ÌQÆ“n‚µ‚ª‚Å‚«‚È‚¢‚Á‚Û‚¢‚Ì‚Å macro ‚Å...
// •¡”‚ÌŒ^‚É‘Î‰‚Å‚«‚é‚Ì‚Å‚Ş‚µ‚ë‚±‚Á‚¿‚Ì‚ª‚¢‚¢‚Ì‚©‚à
#define get_pos(p)      {p.x, p.y, p.z}
#define get_vel(p)      {p.vx, p.vy, p.vz}
#define get_normal(p)   {p.nx, p.ny, p.nz}
#define get_accel(p)    {p.ax, p.ay, p.az}
#define set_pos(p, v)   p.x=v.x; p.y=v.y; p.z=v.z;
#define set_vel(p, v)   p.vx=v.x; p.vy=v.y; p.vz=v.z;
#define set_accel(p, v) p.ax=v.x; p.ay=v.y; p.az=v.z;

#define get_bl(p)       {p.bl_x, p.bl_y, p.bl_z}
#define get_ur(p)       {p.ur_x, p.ur_y, p.ur_z}


#endif // __SPH_collision_h__
