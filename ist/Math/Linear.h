#ifndef ist_Graphic_Linear_h
#define ist_Graphic_Linear_h

#include "ist/Config.h"


namespace ist {

struct Rect
{
    vec2 pos, size;

    Rect() {}
    Rect(const vec2 &p, const vec2 &s) : pos(p), size(s) {}

    const vec2& getPosition() const { return pos; }
    const vec2& getSize() const     { return size; }

    vec2 getUpperRight() const  { return size+pos; }
    vec2 getUpperLeft() const   { return vec2(pos.x, pos.y+size.y); }
    vec2 getBottomLeft() const  { return pos; }
    vec2 getBottomRight() const { return vec2(pos.x+size.x, pos.y); }
};

struct Line2
{
    vec2 pos[2];

    Line2() {}
    Line2(const vec2 &p1, const vec2 &p2) { pos[0]=p1; pos[1]=p2; }

    vec4& operator[](size_t i) { return pos[i]; }
    const vec4& operator[](size_t i) const { return pos[i]; }
};

struct Circle
{
    vec2 pos; float32 radius;

    Circle() : radius(0.0f) {}
    Circle(const vec2 &p, float32 r) : pos(p), radius(r) {}
};


struct istAlign(16) Sphere
{
    vec4 v;

    Sphere() {}
    Sphere(float x, float y, float z, float r) { v=vec4(x, y, z, r) }
    Sphere(vec3 pos, float r) { v=vec4(pos, r); }

    vec4 getPosition() const { return vec4(v.x, v.y, v.z, 0.0f); }
    float getRadius() const { return v.w; }
};


struct istAlign(16) Line
{
    vec4 pos[2];

    Line() {}
    Line(const vec4 &p1, const vec4 &p2) { pos[0]=p1; pos[1]=p2; }

    vec4& operator[](size_t i) { return pos[i]; }
    const vec4& operator[](size_t i) const { return pos[i]; }
};


struct istAlign(16) Capsule
{
    Line segment;
    float32 radius;

    Capsule() {}
    Capsule(const vec4 &p1, const vec4 &p2, float32 r) : segment(p1,p2), radius(r) {}

    vec4& operator[](size_t i) { return segment[i]; }
    const vec4& operator[](size_t i) const { return segment[i]; }
    float getRadius() const { return begin_pos.getRadius(); }
};


struct istAlign(16) Plane
{
    vec4 v;

    Plane() {}
    Plane(const vec3 &n, float d) { v=vec4(n,d); }
};


struct istAlign(16) Triangle
{
    vec4 pos[3];

    Triangle() {}
    Triangle(const vec4 &p0, const vec4 &p1, const vec4 &p2) { pos[0]=p0; pos[1]=p1; pos[2]=p2; }

    vec4& operator[](size_t i) { return pos[i]; }
    const vec4& operator[](size_t i) const { return pos[i]; }
};


struct istAlign(16) AABB
{
    vec4 pos[2];

    AABB() {}
    AABB(const vec4 &_r, const vec4 &bl) { pos[0]=ur; pos[1]=bl; }
};

struct istAlign(16) OBB
{
    AABB aabb;
    mat4 mat;

    OBB() {}
    OBB(const vec4 &ur, const vec4 &bl, const mat4 &m) : AABB(ur, bl), mat(m) {}
    OBB(const AABB &a, const mat4 &m) : aabb(a), mat(m) {}
};


struct istAlign(16) Frustum
{
    mat4 mat;

    Frustum() {}
};



} // namespace ist
#endif // ist_Graphic_Linear_h
